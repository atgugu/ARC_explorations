"""
Test the distribute_objects fix on a single task
"""

import numpy as np
from advanced_solver import AdvancedARCSolver
from arc_generative_solver import evaluate_predictions

# Create the distribute_objects task
task = {
    "train": [{"input": [[1, 0, 0, 0, 0, 2]],
              "output": [[1, 0, 0, 0, 2, 0]]}],
    "test": [{"input": [[3, 0, 0, 0, 0, 4]],
             "output": [[3, 0, 0, 0, 4, 0]]}]
}

print("="*70)
print("TESTING DISTRIBUTE_OBJECTS FIX")
print("="*70)

print("\nTask:")
print(f"  Train input:  {task['train'][0]['input'][0]}")
print(f"  Train output: {task['train'][0]['output'][0]}")
print(f"  Test input:   {task['test'][0]['input'][0]}")
print(f"  Expected:     {task['test'][0]['output'][0]}")

print("\nSolving...")
solver = AdvancedARCSolver(max_candidates=150, beam_width=20, active_inference_steps=5)

pred1, pred2, metadata = solver.solve(task)
target = np.array(task["test"][0]["output"])

eval_res = evaluate_predictions(pred1, pred2, target)

print("\nResults:")
print(f"  Prediction 1: {pred1.tolist()[0]}")
print(f"  Prediction 2: {pred2.tolist()[0]}")
print(f"  Target:       {target.tolist()[0]}")

print("\nEvaluation:")
print(f"  Any correct:   {eval_res['any_correct']}")
print(f"  Accuracy 1:    {eval_res['pixel_accuracy_1']:.1%}")
print(f"  Accuracy 2:    {eval_res['pixel_accuracy_2']:.1%}")
print(f"  Best accuracy: {max(eval_res['pixel_accuracy_1'], eval_res['pixel_accuracy_2']):.1%}")

if eval_res['any_correct']:
    print("\n✓ SUCCESS: distribute_objects is now working correctly!")
else:
    print(f"\n✗ FAILED: Best accuracy {max(eval_res['pixel_accuracy_1'], eval_res['pixel_accuracy_2']):.1%}")

if metadata.get('top_programs'):
    print("\nTop programs used:")
    for i, prog in enumerate(metadata['top_programs'][:3], 1):
        schema = prog.get('schema', 'unknown')
        score = prog.get('score', prog.get('log_prob', 0.0))
        print(f"  {i}. {schema} (score: {score:.3f})")

print("\n" + "="*70)
