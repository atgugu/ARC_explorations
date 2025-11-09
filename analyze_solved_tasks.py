"""Analyze the solved tasks in detail."""

import json
import numpy as np
from arc_curiosity_solver.solver_diverse import DiverseARCCuriositySolver

# Load the solved tasks
tasks_to_analyze = ['25ff71a9', '3c9b0459']

for task_id in tasks_to_analyze:
    print("="*80)
    print(f"ANALYZING SOLVED TASK: {task_id}")
    print("="*80)

    # Load task data
    with open(f'ARC-AGI/data/training/{task_id}.json', 'r') as f:
        task_data = json.load(f)

    train_pairs = [
        (np.array(pair['input']), np.array(pair['output']))
        for pair in task_data['train']
    ]
    test_input = np.array(task_data['test'][0]['input'])
    test_output = np.array(task_data['test'][0]['output'])

    print(f"\nTrain pairs: {len(train_pairs)}")
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")

    # Show training examples
    print(f"\nTraining examples:")
    for i, (inp, out) in enumerate(train_pairs, 1):
        print(f"  Example {i}: {inp.shape} → {out.shape}")
        print(f"    Input unique values: {sorted(set(inp.flatten()))}")
        print(f"    Output unique values: {sorted(set(out.flatten()))}")

    # Solve with verbose mode
    print(f"\n{'='*60}")
    print("SOLVING WITH VERBOSE MODE:")
    print(f"{'='*60}\n")

    solver = DiverseARCCuriositySolver()
    pred1, pred2 = solver.solve(train_pairs, test_input, verbose=True)

    # Check which matched
    match1 = np.array_equal(pred1, test_output)
    match2 = np.array_equal(pred2, test_output)

    print(f"\n{'='*60}")
    print("RESULT:")
    print(f"{'='*60}")
    print(f"Prediction 1: {'✅ EXACT MATCH' if match1 else '❌ No match'}")
    print(f"Prediction 2: {'✅ EXACT MATCH' if match2 else '❌ No match'}")

    # Get pattern info
    if hasattr(solver, 'get_pattern_info'):
        pattern_info = solver.get_pattern_info()
        print(f"\nPattern info:")
        print(f"  Used pattern inference: {pattern_info.get('used_pattern_inference', False)}")
        print(f"  Patterns detected: {pattern_info.get('num_patterns', 0)}")
        if pattern_info.get('patterns'):
            for p in pattern_info['patterns']:
                print(f"    - {p['description']} (confidence={p['confidence']:.2f})")

    print("\n")
