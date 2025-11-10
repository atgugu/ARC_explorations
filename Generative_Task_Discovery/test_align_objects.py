"""
Test align_objects to understand the regression
"""

import numpy as np
from advanced_solver import AdvancedARCSolver
from arc_generative_solver import evaluate_predictions

# Create the align_objects task (from final_comparison.py line 49-58)
task = {
    "train": [{"input": [[1, 0], [0, 0], [0, 2]],
              "output": [[1, 2], [0, 0], [0, 0]]}],
    "test": [{"input": [[3, 0], [0, 0], [0, 0], [4, 0]],
             "output": [[3, 4], [0, 0], [0, 0], [0, 0]]}]
}

print("="*70)
print("TESTING ALIGN_OBJECTS")
print("="*70)

print("\nTask:")
print("  Train input:")
for row in task['train'][0]['input']:
    print(f"    {row}")
print("  Train output:")
for row in task['train'][0]['output']:
    print(f"    {row}")
print("\n  Test input:")
for row in task['test'][0]['input']:
    print(f"    {row}")
print("  Expected output:")
for row in task['test'][0]['output']:
    print(f"    {row}")

print("\nSolving...")
solver = AdvancedARCSolver(max_candidates=150, beam_width=20, active_inference_steps=5)

pred1, pred2, metadata = solver.solve(task)
target = np.array(task["test"][0]["output"])

eval_res = evaluate_predictions(pred1, pred2, target)

print("\nResults:")
print("  Prediction 1:")
for row in pred1.tolist():
    print(f"    {row}")
print("  Prediction 2:")
for row in pred2.tolist():
    print(f"    {row}")
print("  Target:")
for row in target.tolist():
    print(f"    {row}")

print("\nEvaluation:")
print(f"  Any correct:   {eval_res['any_correct']}")
print(f"  Accuracy 1:    {eval_res['pixel_accuracy_1']:.1%}")
print(f"  Accuracy 2:    {eval_res['pixel_accuracy_2']:.1%}")
print(f"  Best accuracy: {max(eval_res['pixel_accuracy_1'], eval_res['pixel_accuracy_2']):.1%}")

if eval_res['any_correct']:
    print("\n✓ SUCCESS: align_objects is working")
else:
    print(f"\n✗ FAILED: Best accuracy {max(eval_res['pixel_accuracy_1'], eval_res['pixel_accuracy_2']):.1%}")
    print("\nDifference analysis:")
    diff1 = pred1 != target
    diff2 = pred2 != target
    print(f"  Prediction 1 errors: {np.sum(diff1)} pixels")
    print(f"  Prediction 2 errors: {np.sum(diff2)} pixels")

if metadata.get('top_programs'):
    print("\nTop programs used:")
    for i, prog in enumerate(metadata['top_programs'][:5], 1):
        schema = prog.get('schema', 'unknown')
        score = prog.get('score', prog.get('log_prob', 0.0))
        print(f"  {i}. {schema}")

print("\n" + "="*70)
