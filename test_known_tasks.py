"""Test on known-solved tasks to verify solver functionality"""

import json
import numpy as np
from pathlib import Path

from arc_curiosity_solver.solver_diverse import DiverseARCCuriositySolver
from arc_curiosity_solver.solver_conditional import ConditionalARCCuriositySolver


def test_known_task(task_id):
    """Test a specific known task."""
    task_file = Path(f"ARC-AGI/data/training/{task_id}.json")

    if not task_file.exists():
        print(f"Task {task_id} not found")
        return

    with open(task_file, 'r') as f:
        task_data = json.load(f)

    test_input = np.array(task_data['test'][0]['input'])
    expected = np.array(task_data['test'][0]['output'])

    train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                  for ex in task_data['train']]

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    # Test diverse solver
    print("\n1. DIVERSE SOLVER")
    print("-"*60)
    diverse = DiverseARCCuriositySolver()
    diverse.verbose = False
    div_hyps = diverse._generate_hypotheses(train_pairs, test_input)

    print(f"Generated {len(div_hyps)} hypotheses")

    for i, h in enumerate(div_hyps[:5], 1):
        try:
            pred = h.program.function(test_input.copy())
            match = np.array_equal(pred, expected)
            acc = (pred == expected).mean() * 100 if pred.shape == expected.shape else 0
            variant = h.parameters.get('variant', '?')
            desc = h.parameters.get('description', 'no desc')[:40]

            symbol = "✓" if match else " "
            print(f"  {i}. [{variant:10}] {symbol} {acc:5.1f}% - {desc}")
        except Exception as e:
            print(f"  {i}. ERROR: {str(e)[:50]}")

    # Test conditional solver
    print("\n2. CONDITIONAL SOLVER")
    print("-"*60)
    conditional = ConditionalARCCuriositySolver()
    conditional.verbose = False
    cond_hyps = conditional._generate_hypotheses(train_pairs, test_input)

    print(f"Generated {len(cond_hyps)} hypotheses")

    for i, h in enumerate(cond_hyps[:5], 1):
        try:
            pred = h.program.function(test_input.copy())
            match = np.array_equal(pred, expected)
            acc = (pred == expected).mean() * 100 if pred.shape == expected.shape else 0
            variant = h.parameters.get('variant', '?')
            desc = h.parameters.get('description', 'no desc')[:40]

            symbol = "✓" if match else " "
            print(f"  {i}. [{variant:10}] {symbol} {acc:5.1f}% - {desc}")
        except Exception as e:
            print(f"  {i}. ERROR: {str(e)[:50]}")


if __name__ == '__main__':
    # Test known-solved tasks
    print("="*60)
    print("TESTING KNOWN-SOLVED TASKS")
    print("="*60)

    test_known_task('25ff71a9')  # Previously solved by diverse
    test_known_task('3c9b0459')  # Previously solved by diverse

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
