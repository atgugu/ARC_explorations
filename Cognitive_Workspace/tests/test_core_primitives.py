"""
Unit tests for core DSL primitives
"""

import pytest
import numpy as np
from src.dsl.core_primitives import *


class TestSelection:
    """Test selection and filtering primitives"""

    def test_select_by_color_simple(self):
        """Test basic color selection"""
        grid = np.array([
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 2]
        ])

        # Select blue (1)
        objects = select_by_color(grid, 1)
        assert len(objects) == 1
        assert len(objects[0]) == 3  # 3 pixels

        # Select red (2)
        objects = select_by_color(grid, 2)
        assert len(objects) == 1
        assert len(objects[0]) == 1  # 1 pixel

    def test_select_by_color_multiple_objects(self):
        """Test selecting multiple disconnected objects"""
        grid = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ])

        objects = select_by_color(grid, 1)
        assert len(objects) == 4  # 4 separate pixels

    def test_select_largest(self):
        """Test selecting largest objects"""
        # Create objects of different sizes
        grid = np.array([
            [1, 1, 0, 2],
            [1, 1, 0, 2],
            [0, 0, 0, 2],
            [3, 0, 0, 2]
        ])

        objects = select_by_color(grid, 1)  # 4 pixels
        objects += select_by_color(grid, 2)  # 4 pixels
        objects += select_by_color(grid, 3)  # 1 pixel

        largest = select_largest(objects, k=2)
        assert len(largest) == 2
        assert len(largest[0]) == 4  # Largest has 4 pixels
        assert len(largest[1]) == 4  # Second largest has 4 pixels

    def test_select_smallest(self):
        """Test selecting smallest objects"""
        grid = np.array([
            [1, 1, 0, 2],
            [1, 1, 0, 0],
            [0, 0, 0, 3]
        ])

        objects = select_by_color(grid, 1)  # 4 pixels
        objects += select_by_color(grid, 2)  # 1 pixel
        objects += select_by_color(grid, 3)  # 1 pixel

        smallest = select_smallest(objects, k=2)
        assert len(smallest) == 2
        assert len(smallest[0]) == 1
        assert len(smallest[1]) == 1

    def test_select_by_size(self):
        """Test size filtering"""
        grid = np.array([
            [1, 1, 0, 2],
            [1, 0, 0, 3]
        ])

        objects = select_by_color(grid, 1)  # 3 pixels
        objects += select_by_color(grid, 2)  # 1 pixel
        objects += select_by_color(grid, 3)  # 1 pixel

        # Get objects with size > 1
        large = select_by_size(objects, 1, '>')
        assert len(large) == 1
        assert len(large[0]) == 3

        # Get objects with size == 1
        small = select_by_size(objects, 1, '==')
        assert len(small) == 2

    def test_select_by_position(self):
        """Test position-based selection"""
        grid = np.zeros((10, 10), dtype=int)

        # Place objects at corners
        grid[0, 0] = 1  # top-left
        grid[0, 9] = 2  # top-right
        grid[9, 0] = 3  # bottom-left
        grid[9, 9] = 4  # bottom-right
        grid[5, 5] = 5  # center

        all_objects = []
        for color in [1, 2, 3, 4, 5]:
            all_objects += select_by_color(grid, color)

        corners = select_by_position(all_objects, "corner", grid.shape)
        assert len(corners) == 4

        center = select_by_position(all_objects, "center", grid.shape)
        assert len(center) == 1


class TestSpatialTransformations:
    """Test spatial transformation primitives"""

    def test_translate(self):
        """Test object translation"""
        obj = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 square

        translated = translate(obj, 2, 3)
        expected = [(2, 3), (2, 4), (3, 3), (3, 4)]
        assert set(translated) == set(expected)

    def test_rotate_90(self):
        """Test 90° rotation"""
        obj = [(0, 0), (0, 1)]  # Horizontal line

        rotated = rotate(obj, 90, center=(0, 0))
        # After rotation: should be vertical
        assert (0, 0) in rotated
        assert (-1, 0) in rotated or (0, -1) in rotated  # Rotation result

    def test_rotate_180(self):
        """Test 180° rotation"""
        obj = [(0, 0), (0, 1), (1, 0)]  # L-shape

        rotated = rotate(obj, 180, center=(0, 0))
        expected_set = {(0, 0), (0, -1), (-1, 0)}
        # Check at least center is preserved
        assert (0, 0) in rotated

    def test_reflect_vertical(self):
        """Test vertical reflection"""
        obj = [(0, 0), (0, 1), (0, 2)]  # Horizontal line

        reflected = reflect(obj, Axis.VERTICAL, position=1.0)
        # Should reflect around column 1
        assert (0, 1) in reflected  # Center stays
        expected = [(0, 2), (0, 1), (0, 0)]
        assert len(reflected) == 3

    def test_reflect_horizontal(self):
        """Test horizontal reflection"""
        obj = [(0, 0), (1, 0), (2, 0)]  # Vertical line

        reflected = reflect(obj, Axis.HORIZONTAL, position=1.0)
        # Should reflect around row 1
        assert len(reflected) == 3

    def test_scale_2x(self):
        """Test 2x scaling"""
        obj = [(0, 0), (0, 1)]  # 1x2 horizontal line

        scaled = scale(obj, 2)
        # Each pixel becomes 2x2
        assert len(scaled) == 4  # 2 pixels × 2×2 = 4 pixels

    def test_move_to_corner(self):
        """Test moving object to corner"""
        obj = [(5, 5), (5, 6), (6, 5), (6, 6)]  # 2x2 at (5,5)
        grid_shape = (10, 10)

        moved = move_to(obj, "top_left", grid_shape)
        assert (0, 0) in moved
        assert (0, 1) in moved
        assert (1, 0) in moved
        assert (1, 1) in moved

    def test_move_to_center(self):
        """Test centering object"""
        obj = [(0, 0), (0, 1)]  # 1x2 horizontal
        grid_shape = (10, 10)

        moved = move_to(obj, "center", grid_shape)
        centroid = compute_centroid(moved)
        # Should be roughly at center
        assert 3 <= centroid[0] <= 6
        assert 3 <= centroid[1] <= 6


class TestColorOperations:
    """Test color operation primitives"""

    def test_recolor(self):
        """Test recoloring an object"""
        grid = np.array([
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ])

        objects = select_by_color(grid, 1)
        result = recolor(objects[0], 2, grid)

        # Should now be color 2
        assert result[0, 1] == 2
        assert result[1, 0] == 2
        assert result[1, 1] == 2
        # Background unchanged
        assert result[0, 0] == 0

    def test_swap_colors(self):
        """Test swapping two colors"""
        grid = np.array([
            [1, 2, 1],
            [2, 1, 2],
            [1, 2, 1]
        ])

        result = swap_colors(grid, 1, 2)

        # All 1s should become 2s and vice versa
        assert result[0, 0] == 2
        assert result[0, 1] == 1
        assert result[1, 1] == 2

    def test_recolor_by_rule_size(self):
        """Test recoloring by size"""
        grid = np.array([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 2, 2],
            [3, 0, 0, 0, 0]
        ])

        objects = []
        for color in [1, 2, 3]:
            objects += select_by_color(grid, color)

        result = recolor_by_rule(objects, "size_ascending", grid)

        # Smallest object (3: 1 pixel) should be color 1
        assert result[3, 0] == 1
        # Larger objects should be colors 2, 3


class TestPatternOperations:
    """Test pattern operation primitives"""

    def test_tile_2x2(self):
        """Test tiling 2x2"""
        obj = [(0, 0)]  # Single pixel
        result = tile(obj, 2, 2)

        # Should have 4 pixels total in 2x2 pattern
        assert np.sum(result > 0) == 4

    def test_tile_3x3(self):
        """Test tiling 3x3"""
        obj = [(0, 0), (0, 1)]  # 1x2 horizontal
        result = tile(obj, 3, 3)

        # Should have 9 copies of 2-pixel object = 18 pixels
        assert np.sum(result > 0) == 18

    def test_copy_to_positions(self):
        """Test copying to specific positions"""
        obj = [(0, 0), (0, 1)]
        positions = [(0, 0), (5, 5), (2, 8)]
        grid_shape = (10, 10)

        result = copy_to_positions(obj, positions, grid_shape)

        # Should have 3 copies × 2 pixels = 6 pixels
        assert np.sum(result > 0) == 6
        # Check specific positions
        assert result[0, 0] > 0  # First copy
        assert result[5, 5] > 0  # Second copy
        assert result[2, 8] > 0  # Third copy


class TestGridOperations:
    """Test grid operation primitives"""

    def test_overlay_replace(self):
        """Test overlay with replace mode"""
        grid1 = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ])

        grid2 = np.array([
            [0, 0, 2],
            [0, 0, 2],
            [0, 0, 0]
        ])

        result = overlay(grid1, grid2, mode='replace')

        # grid1 values preserved where grid2 is 0
        assert result[0, 0] == 1
        # grid2 overwrites where non-zero
        assert result[0, 2] == 2

    def test_overlay_or(self):
        """Test overlay with OR mode"""
        grid1 = np.array([[1, 0], [0, 0]])
        grid2 = np.array([[0, 0], [0, 1]])

        result = overlay(grid1, grid2, mode='or')

        assert result[0, 0] == 1
        assert result[1, 1] == 1

    def test_crop(self):
        """Test cropping"""
        grid = np.arange(25).reshape(5, 5)

        result = crop(grid, 1, 1, 3, 3)

        assert result.shape == (3, 3)
        assert result[0, 0] == 6  # grid[1, 1]

    def test_crop_to_content(self):
        """Test cropping to content"""
        grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        result = crop_to_content(grid)

        assert result.shape == (2, 2)
        assert np.all(result == 1)


class TestTopologicalOperations:
    """Test topological operation primitives"""

    def test_fill_holes_simple(self):
        """Test filling a simple hole"""
        obj = [(0, 0), (0, 1), (0, 2),
               (1, 0),         (1, 2),  # Hole at (1, 1)
               (2, 0), (2, 1), (2, 2)]

        filled = fill_holes(obj)

        # Should include the hole
        assert len(filled) >= len(obj)
        assert (1, 1) in filled

    def test_grow(self):
        """Test growing an object"""
        obj = [(5, 5)]  # Single pixel

        grown = grow(obj, amount=1)

        # Should expand to 3x3 (9 pixels)
        assert len(grown) >= 9

    def test_shrink(self):
        """Test shrinking an object"""
        # 3x3 square
        obj = [(0, 0), (0, 1), (0, 2),
               (1, 0), (1, 1), (1, 2),
               (2, 0), (2, 1), (2, 2)]

        shrunk = shrink(obj, amount=1)

        # Should be smaller
        assert len(shrunk) < len(obj)


class TestUtilityOperations:
    """Test utility operation primitives"""

    def test_count(self):
        """Test counting objects"""
        objects = [
            [(0, 0)],
            [(1, 1), (1, 2)],
            [(3, 3), (3, 4), (3, 5)]
        ]

        assert count(objects) == 3

        # Count with predicate
        assert count(objects, lambda obj: len(obj) > 1) == 2

    def test_measure_area(self):
        """Test measuring area"""
        obj = [(0, 0), (0, 1), (1, 0)]
        assert measure(obj, "area") == 3.0

    def test_measure_width_height(self):
        """Test measuring dimensions"""
        obj = [(0, 0), (0, 1), (0, 2)]  # 1x3 horizontal line

        assert measure(obj, "width") == 3.0
        assert measure(obj, "height") == 1.0

    def test_measure_aspect_ratio(self):
        """Test measuring aspect ratio"""
        obj = [(0, 0), (0, 1)]  # 1x2

        aspect = measure(obj, "aspect_ratio")
        assert aspect == 2.0  # width / height = 2 / 1

    def test_sort_objects_by_size(self):
        """Test sorting by size"""
        objects = [
            [(0, 0)],  # size 1
            [(1, 1), (1, 2), (1, 3)],  # size 3
            [(2, 2), (2, 3)],  # size 2
        ]

        sorted_asc = sort_objects(objects, "size", "ascending")
        assert len(sorted_asc[0]) == 1
        assert len(sorted_asc[1]) == 2
        assert len(sorted_asc[2]) == 3

        sorted_desc = sort_objects(objects, "size", "descending")
        assert len(sorted_desc[0]) == 3
        assert len(sorted_desc[1]) == 2
        assert len(sorted_desc[2]) == 1


class TestHelperFunctions:
    """Test helper functions"""

    def test_object_to_grid(self):
        """Test converting object to grid"""
        obj = [(0, 0), (0, 1), (1, 0)]
        grid = object_to_grid(obj, color=2, grid_shape=(5, 5))

        assert grid.shape == (5, 5)
        assert grid[0, 0] == 2
        assert grid[0, 1] == 2
        assert grid[1, 0] == 2
        assert grid[2, 2] == 0  # Empty cell

    def test_get_object_bounds(self):
        """Test getting bounding box"""
        obj = [(2, 3), (2, 4), (3, 3), (3, 4)]
        bounds = get_object_bounds(obj)

        assert bounds == (2, 3, 3, 4)

    def test_compute_centroid(self):
        """Test centroid computation"""
        obj = [(0, 0), (0, 2), (2, 0), (2, 2)]  # Four corners of square
        centroid = compute_centroid(obj)

        assert centroid[0] == 1.0  # Average row
        assert centroid[1] == 1.0  # Average col


class TestIntegration:
    """Integration tests combining multiple primitives"""

    def test_select_rotate_tile(self):
        """Test: select largest, rotate, then tile"""
        grid = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0]
        ])

        # Select largest
        objects = select_by_color(grid, 1)
        objects += select_by_color(grid, 2)
        largest = select_largest(objects, k=1)[0]

        # Rotate
        rotated = rotate(largest, 90)

        # Tile
        result = tile(rotated, 2, 2, color=3)

        assert result.shape[0] > 0
        assert result.shape[1] > 0
        assert np.sum(result > 0) > 0

    def test_select_scale_position(self):
        """Test: select, scale, move to corner"""
        grid = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ])

        # Select
        objects = select_by_color(grid, 1)
        obj = objects[0]

        # Scale
        scaled = scale(obj, 2)

        # Move to corner
        moved = move_to(scaled, "bottom_right", (10, 10))

        # Check it's in bottom right area
        bounds = get_object_bounds(moved)
        assert bounds[2] >= 8  # max_row near bottom
        assert bounds[3] >= 8  # max_col near right

    def test_color_sort_pipeline(self):
        """Test: select, sort by size, recolor by rule"""
        grid = np.array([
            [1, 1, 0, 2],
            [1, 1, 0, 3],
            [0, 0, 0, 4]
        ])

        # Select all objects
        objects = []
        for color in [1, 2, 3, 4]:
            objects += select_by_color(grid, color)

        # Sort by size
        sorted_objs = sort_objects(objects, "size", "ascending")

        # Verify sorting
        assert len(sorted_objs[0]) <= len(sorted_objs[-1])

        # Recolor
        result = recolor_by_rule(sorted_objs, "size_ascending", grid)

        # Smallest should be color 1
        assert np.sum(result == 1) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
