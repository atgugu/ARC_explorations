"""Test single task with diverse solver in verbose mode."""

import json
import numpy as np
from arc_curiosity_solver.solver_diverse import DiverseARCCuriositySolver

# Load task 150deff5 (86.4% with patterns)
with open('ARC-AGI/data/training/150deff5.json', 'r') as f:
    task_data = json.load(f)

train_pairs = [
    (np.array(pair['input']), np.array(pair['output']))
    for pair in task_data['train']
]
test_input = np.array(task_data['test'][0]['input'])
test_output = np.array(task_data['test'][0]['output'])

print("="*80)
print("TESTING DIVERSE SOLVER ON TASK 150deff5")
print("="*80)
print(f"Train pairs: {len(train_pairs)}")
print(f"Test input shape: {test_input.shape}")
print(f"Test output shape: {test_output.shape}")
print()

solver = DiverseARCCuriositySolver()
pred1, pred2 = solver.solve(train_pairs, test_input, verbose=True)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

match1 = np.array_equal(pred1, test_output)
match2 = np.array_equal(pred2, test_output)

if pred1.shape == test_output.shape:
    acc1 = np.mean(pred1 == test_output)
else:
    acc1 = 0.0

if pred2.shape == test_output.shape:
    acc2 = np.mean(pred2 == test_output)
else:
    acc2 = 0.0

print(f"Prediction 1: {'✅ EXACT MATCH' if match1 else f'{acc1*100:.1f}% accurate'}")
print(f"Prediction 2: {'✅ EXACT MATCH' if match2 else f'{acc2*100:.1f}% accurate'}")

if not (match1 or match2):
    print(f"\n❌ No exact match. Best: {max(acc1, acc2)*100:.1f}%")

    # Show where predictions differ
    best_pred = pred1 if acc1 >= acc2 else pred2
    if best_pred.shape == test_output.shape:
        diff = best_pred != test_output
        print(f"Pixels different: {diff.sum()} / {test_output.size}")
        print(f"Accuracy: {(1 - diff.sum() / test_output.size)*100:.2f}%")
