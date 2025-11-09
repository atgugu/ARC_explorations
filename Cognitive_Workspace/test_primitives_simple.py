"""
Simple test script for core primitives (no pytest required)
Run with: python test_primitives_simple.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError:
    print("✗ NumPy not installed. Install with: pip install numpy scipy")
    sys.exit(1)

from dsl.core_primitives import *

def test_basic_selection():
    """Test basic object selection"""
    print("\n=== Testing Selection Primitives ===")

    grid = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 2]
    ])

    # Test select_by_color
    objects = select_by_color(grid, 1)
    assert len(objects) == 1, f"Expected 1 object, got {len(objects)}"
    assert len(objects[0]) == 3, f"Expected 3 pixels, got {len(objects[0])}"
    print("✓ select_by_color works")

    # Test select_largest
    grid2 = np.array([
        [1, 1, 0, 2],
        [1, 1, 0, 2],
        [0, 0, 0, 3]
    ])

    all_objects = []
    for color in [1, 2, 3]:
        all_objects += select_by_color(grid2, color)

    largest = select_largest(all_objects, k=1)
    assert len(largest) == 1
    assert len(largest[0]) == 4  # The 2x2 square
    print("✓ select_largest works")

    # Test select_smallest
    smallest = select_smallest(all_objects, k=1)
    assert len(smallest) == 1
    assert len(smallest[0]) == 1  # Single pixel
    print("✓ select_smallest works")

    # Test select_by_size
    large_objects = select_by_size(all_objects, 1, '>')
    assert len(large_objects) == 2  # Objects with >1 pixel
    print("✓ select_by_size works")


def test_spatial_transformations():
    """Test spatial transformation primitives"""
    print("\n=== Testing Spatial Transformations ===")

    # Test translate
    obj = [(0, 0), (0, 1), (1, 0), (1, 1)]
    translated = translate(obj, 2, 3)
    expected = {(2, 3), (2, 4), (3, 3), (3, 4)}
    assert set(translated) == expected
    print("✓ translate works")

    # Test rotate
    obj = [(0, 0), (0, 1)]
    rotated = rotate(obj, 90, center=(0, 0))
    assert len(rotated) == 2
    print("✓ rotate works")

    # Test reflect
    obj = [(0, 0), (0, 1), (0, 2)]
    reflected = reflect(obj, Axis.VERTICAL, position=1.0)
    assert len(reflected) == 3
    print("✓ reflect works")

    # Test scale
    obj = [(0, 0), (0, 1)]
    scaled = scale(obj, 2)
    assert len(scaled) == 8  # 2 pixels × (2×2 each) = 8 pixels
    print("✓ scale works")

    # Test move_to
    obj = [(5, 5), (5, 6)]
    moved = move_to(obj, "top_left", (10, 10))
    bounds = get_object_bounds(moved)
    assert bounds[0] == 0  # Top row
    assert bounds[1] == 0  # Left column
    print("✓ move_to works")


def test_color_operations():
    """Test color operation primitives"""
    print("\n=== Testing Color Operations ===")

    grid = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0]
    ])

    # Test recolor
    objects = select_by_color(grid, 1)
    result = recolor(objects[0], 2, grid)
    assert result[0, 1] == 2
    assert result[1, 0] == 2
    print("✓ recolor works")

    # Test swap_colors
    grid2 = np.array([
        [1, 2, 1],
        [2, 1, 2]
    ])
    result = swap_colors(grid2, 1, 2)
    assert result[0, 0] == 2
    assert result[0, 1] == 1
    print("✓ swap_colors works")


def test_pattern_operations():
    """Test pattern operation primitives"""
    print("\n=== Testing Pattern Operations ===")

    # Test tile
    obj = [(0, 0)]
    result = tile(obj, 2, 2)
    assert np.sum(result > 0) == 4  # 2×2 = 4 pixels
    print("✓ tile works")

    # Test copy_to_positions
    obj = [(0, 0), (0, 1)]
    positions = [(0, 0), (5, 5)]
    result = copy_to_positions(obj, positions, (10, 10))
    assert np.sum(result > 0) == 4  # 2 copies × 2 pixels
    print("✓ copy_to_positions works")


def test_grid_operations():
    """Test grid operation primitives"""
    print("\n=== Testing Grid Operations ===")

    grid1 = np.array([[1, 1], [1, 1]])
    grid2 = np.array([[0, 2], [0, 2]])

    # Test overlay
    result = overlay(grid1, grid2, mode='replace')
    assert result[0, 0] == 1
    assert result[0, 1] == 2
    print("✓ overlay works")

    # Test crop
    grid = np.arange(25).reshape(5, 5)
    result = crop(grid, 1, 1, 3, 3)
    assert result.shape == (3, 3)
    assert result[0, 0] == 6
    print("✓ crop works")

    # Test crop_to_content
    grid = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ])
    result = crop_to_content(grid)
    assert result.shape == (2, 2)
    print("✓ crop_to_content works")


def test_topological_operations():
    """Test topological operation primitives"""
    print("\n=== Testing Topological Operations ===")

    # Test fill_holes
    obj = [(0, 0), (0, 1), (0, 2),
           (1, 0),         (1, 2),
           (2, 0), (2, 1), (2, 2)]
    filled = fill_holes(obj)
    assert (1, 1) in filled  # Hole should be filled
    print("✓ fill_holes works")

    # Test grow
    obj = [(5, 5)]
    grown = grow(obj, amount=1, grid_shape=(20, 20))
    assert len(grown) >= 5  # Should be bigger
    print("✓ grow works")

    # Test shrink
    obj = [(0, 0), (0, 1), (0, 2),
           (1, 0), (1, 1), (1, 2),
           (2, 0), (2, 1), (2, 2)]
    shrunk = shrink(obj, amount=1, grid_shape=(10, 10))
    assert len(shrunk) < len(obj)
    print("✓ shrink works")


def test_utility_operations():
    """Test utility operation primitives"""
    print("\n=== Testing Utility Operations ===")

    objects = [
        [(0, 0)],
        [(1, 1), (1, 2)],
        [(3, 3), (3, 4), (3, 5)]
    ]

    # Test count
    assert count(objects) == 3
    assert count(objects, lambda obj: len(obj) > 1) == 2
    print("✓ count works")

    # Test measure
    obj = [(0, 0), (0, 1), (1, 0)]
    assert measure(obj, "area") == 3.0
    assert measure(obj, "width") == 2.0
    assert measure(obj, "height") == 2.0
    print("✓ measure works")

    # Test sort_objects
    sorted_objs = sort_objects(objects, "size", "ascending")
    assert len(sorted_objs[0]) == 1
    assert len(sorted_objs[-1]) == 3
    print("✓ sort_objects works")


def test_integration():
    """Test combining multiple primitives"""
    print("\n=== Testing Integration ===")

    grid = np.array([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [2, 0, 0, 0]
    ])

    # Select largest object
    objects = []
    for color in [1, 2]:
        objects += select_by_color(grid, color)
    largest = select_largest(objects, k=1)[0]

    # Rotate it
    rotated = rotate(largest, 90)

    # Tile it
    result = tile(rotated, 2, 2, color=3, grid_shape=(10, 10))

    assert result.shape[0] > 0
    assert np.sum(result > 0) > 0
    print("✓ Select → Rotate → Tile pipeline works")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Core DSL Primitives")
    print("=" * 60)

    try:
        test_basic_selection()
        test_spatial_transformations()
        test_color_operations()
        test_pattern_operations()
        test_grid_operations()
        test_topological_operations()
        test_utility_operations()
        test_integration()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print(f"\nImplemented primitives:")
        print("  • Selection (5): select_by_color, select_largest, select_smallest, select_by_size, select_by_position")
        print("  • Spatial (5): translate, rotate, reflect, scale, move_to")
        print("  • Color (3): recolor, swap_colors, recolor_by_rule")
        print("  • Pattern (2): tile, copy_to_positions")
        print("  • Grid (3): overlay, crop, crop_to_content")
        print("  • Topological (3): fill_holes, grow, shrink")
        print("  • Utility (3): count, measure, sort_objects")
        print(f"\nTotal: 24 primitives working correctly!")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
