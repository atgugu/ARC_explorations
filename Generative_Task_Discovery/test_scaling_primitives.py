"""
Test scaling primitives directly to verify they work
"""

import numpy as np
from advanced_primitives import AdvancedPrimitives
from advanced_solver import AdvancedARCSolver
from arc_generative_solver import evaluate_predictions

# Test upscaling primitive directly
print("="*70)
print("TESTING SCALING PRIMITIVES DIRECTLY")
print("="*70)

# Test 1: Upscale by 2x
print("\nTest 1: Upscale by 2x")
input_grid = np.array([[1, 2], [3, 4]])
expected = np.array([[1, 1, 2, 2],
                     [1, 1, 2, 2],
                     [3, 3, 4, 4],
                     [3, 3, 4, 4]])

adv = AdvancedPrimitives()
result = adv.upscale(input_grid, factor=2)

print(f"  Input shape: {input_grid.shape}")
print(f"  Expected shape: {expected.shape}")
print(f"  Result shape: {result.shape}")
print(f"  Match: {np.array_equal(result, expected)}")

if not np.array_equal(result, expected):
    print("  Expected:")
    print(expected)
    print("  Got:")
    print(result)

# Test 2: Downscale by 2x
print("\nTest 2: Downscale by 2x")
input_grid = expected  # Use the upscaled version
expected_down = np.array([[1, 2], [3, 4]])
result_down = adv.downscale(input_grid, factor=2)

print(f"  Input shape: {input_grid.shape}")
print(f"  Expected shape: {expected_down.shape}")
print(f"  Result shape: {result_down.shape}")
print(f"  Match: {np.array_equal(result_down, expected_down)}")

# Test 3: On actual synthetic task
print("\n" + "="*70)
print("TESTING ON ACTUAL SYNTHETIC TASK")
print("="*70)

# Load one of the double_size tasks
import json
with open("synthetic_evaluation_200.json", "r") as f:
    tasks = json.load(f)

# Find first double_size task
double_task = None
for task in tasks:
    if task["name"].startswith("double_size"):
        double_task = task
        break

if double_task:
    print(f"\nTask: {double_task['name']}")
    train_input = np.array(double_task["task"]["train"][0]["input"])
    train_output = np.array(double_task["task"]["train"][0]["output"])
    test_input = np.array(double_task["task"]["test"][0]["input"])
    test_output = np.array(double_task["task"]["test"][0]["output"])

    print(f"  Train input shape: {train_input.shape}")
    print(f"  Train output shape: {train_output.shape}")
    print(f"  Test input shape: {test_input.shape}")
    print(f"  Test output shape: {test_output.shape}")

    # Check if upscale by 2 produces correct output
    result = adv.upscale(test_input, factor=2)
    print(f"  Upscale(2) result shape: {result.shape}")
    print(f"  Expected shape: {test_output.shape}")

    accuracy = np.mean(result == test_output)
    print(f"  Accuracy: {accuracy:.1%}")

    if accuracy < 1.0:
        print("\n  Train input:")
        print(train_input)
        print("\n  Train output:")
        print(train_output)
        print("\n  Test input:")
        print(test_input)
        print("\n  Test output (expected):")
        print(test_output)
        print("\n  Upscale result:")
        print(result)

# Test 4: Try solver on double_size task
print("\n" + "="*70)
print("TESTING SOLVER ON DOUBLE_SIZE TASK")
print("="*70)

solver = AdvancedARCSolver(max_candidates=150, beam_width=20, active_inference_steps=5)

if double_task:
    task = double_task["task"]
    pred1, pred2, metadata = solver.solve(task)
    target = np.array(task["test"][0]["output"])

    eval_res = evaluate_predictions(pred1, pred2, target)

    print(f"\nTask: {double_task['name']}")
    print(f"  Pred1 shape: {pred1.shape}, accuracy: {eval_res['pixel_accuracy_1']:.1%}")
    print(f"  Pred2 shape: {pred2.shape}, accuracy: {eval_res['pixel_accuracy_2']:.1%}")
    print(f"  Target shape: {target.shape}")
    print(f"  Success: {eval_res['any_correct']}")

    if metadata.get('top_programs'):
        print("\n  Top programs:")
        for i, prog in enumerate(metadata['top_programs'][:5], 1):
            schema = prog.get('schema', 'unknown')
            print(f"    {i}. {schema}")

print("\n" + "="*70)
print("✓ SCALING PRIMITIVE TEST COMPLETE")
print("="*70)
