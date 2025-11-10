"""
Test distribute_objects fix

Analyze the exact behavior expected and verify the fix.
"""

import numpy as np

# Test case from final_comparison.py
train_input = [[1, 0, 0, 0, 0, 2]]
train_output = [[1, 0, 0, 0, 2, 0]]

test_input = [[3, 0, 0, 0, 0, 4]]
test_expected = [[3, 0, 0, 0, 4, 0]]

print("Training example:")
print(f"  Input:  {train_input[0]}")
print(f"  Output: {train_output[0]}")
print(f"  Object 1 position: {train_input[0].index(1)} -> {train_output[0].index(1)}")
print(f"  Object 2 position: {train_input[0].index(2)} -> {train_output[0].index(2)}")

print("\nTest example:")
print(f"  Input:    {test_input[0]}")
print(f"  Expected: {test_expected[0]}")
print(f"  Object 3 position: {test_input[0].index(3)} -> {test_expected[0].index(3)}")
print(f"  Object 4 position: {test_input[0].index(4)} -> {test_expected[0].index(4)}")

# Analysis
w = 6
n_objects = 2
total_width = 2  # Each object has width 1

print("\n" + "="*60)
print("ALGORITHM ANALYSIS")
print("="*60)

print(f"\nGrid width: {w}")
print(f"Number of objects: {n_objects}")
print(f"Total object width: {total_width}")

# Current (wrong) formula
spacing_old = (w - total_width) // (n_objects + 1)
print(f"\nCurrent formula: spacing = (w - total_width) // (n_objects + 1)")
print(f"  = ({w} - {total_width}) // ({n_objects} + 1)")
print(f"  = {spacing_old}")
print(f"  Positions: x=1, x=3 ❌")

# Proposed fix: Equal stride starting from 0
if n_objects > 1:
    stride = (w - 2) // (n_objects - 1)
    positions = [i * stride for i in range(n_objects)]
else:
    positions = [0]

print(f"\nProposed formula: stride = (w - 2) // (n_objects - 1), start at 0")
print(f"  = ({w} - 2) // ({n_objects} - 1)")
print(f"  = {stride}")
print(f"  Positions: {positions}")
print(f"  Match expected: {positions == [0, 4]} ✓")

# Test with 3 objects
print("\n" + "="*60)
print("EXTRAPOLATING TO 3 OBJECTS")
print("="*60)

n_objects = 3
if n_objects > 1:
    stride = (w - 2) // (n_objects - 1)
    positions = [i * stride for i in range(n_objects)]
    print(f"\n3 objects in width {w}:")
    print(f"  stride = (6 - 2) // (3 - 1) = {stride}")
    print(f"  Positions: {positions}")
    print(f"  This seems reasonable: obj1 at 0, obj2 at 2, obj3 at 4")

print("\n" + "="*60)
print("✓ ANALYSIS COMPLETE")
print("="*60)
print("\nConclusion: Change algorithm to:")
print("  1. Place first object at x=0")
print("  2. Calculate stride = (w - 2) // (n_objects - 1)")
print("  3. Place object i at position i * stride")
