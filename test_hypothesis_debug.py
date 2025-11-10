"""Debug hypothesis generation to understand the issue"""

import json
import numpy as np
from pathlib import Path
from arc_curiosity_solver.solver_conditional import ConditionalARCCuriositySolver
from arc_curiosity_solver.solver_diverse import DiverseARCCuriositySolver

def compare_hypotheses():
    # Load a previously solved task
    task_file = Path("ARC-AGI/data/training/25ff71a9.json")
    with open(task_file, 'r') as f:
        task_data = json.load(f)

    train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                  for ex in task_data['train']]
    test_input = np.array(task_data['test'][0]['input'])
    expected = np.array(task_data['test'][0]['output'])

    print("="*70)
    print("HYPOTHESIS GENERATION COMPARISON")
    print("Task: 25ff71a9 (Previously solved by diverse solver)")
    print("="*70)

    # Test diverse solver (baseline)
    print("\n1. DIVERSE SOLVER (Baseline - 100% accurate)")
    print("-"*70)
    diverse_solver = DiverseARCCuriositySolver()
    diverse_solver.verbose = False  # Ensure verbose is set
    diverse_hyps = diverse_solver._generate_hypotheses(train_pairs, test_input)

    print(f"Generated {len(diverse_hyps)} hypotheses:")
    for i, h in enumerate(diverse_hyps[:10], 1):
        variant = h.parameters.get('variant', 'unknown')
        desc = h.parameters.get('description', 'no description')
        print(f"  {i}. [{variant:12}] {h.name:20} (act={h.activation:.3f}) {desc[:40]}")

    # Test top prediction
    try:
        pred1 = diverse_hyps[0].program.function(test_input.copy())
        match1 = np.array_equal(pred1, expected)
        acc1 = (pred1 == expected).mean() * 100 if pred1.shape == expected.shape else 0

        pred2 = diverse_hyps[1].program.function(test_input.copy())
        match2 = np.array_equal(pred2, expected)
        acc2 = (pred2 == expected).mean() * 100 if pred2.shape == expected.shape else 0

        print(f"\nPrediction 1: {acc1:.1f}% {'✓ SOLVED' if match1 else ''}")
        print(f"Prediction 2: {acc2:.1f}% {'✓ SOLVED' if match2 else ''}")
    except Exception as e:
        print(f"Error testing predictions: {e}")

    # Test conditional solver
    print("\n" + "="*70)
    print("2. CONDITIONAL SOLVER (With spatial predicates)")
    print("-"*70)
    conditional_solver = ConditionalARCCuriositySolver()
    conditional_solver.verbose = False  # Ensure verbose is set
    conditional_hyps = conditional_solver._generate_hypotheses(train_pairs, test_input)

    print(f"Generated {len(conditional_hyps)} hypotheses:")
    for i, h in enumerate(conditional_hyps[:10], 1):
        variant = h.parameters.get('variant', 'unknown')
        desc = h.parameters.get('description', 'no description')
        print(f"  {i}. [{variant:12}] {h.name:20} (act={h.activation:.3f}) {desc[:40]}")

    # Test top prediction
    try:
        pred1 = conditional_hyps[0].program.function(test_input.copy())
        match1 = np.array_equal(pred1, expected)
        acc1 = (pred1 == expected).mean() * 100 if pred1.shape == expected.shape else 0

        pred2 = conditional_hyps[1].program.function(test_input.copy())
        match2 = np.array_equal(pred2, expected)
        acc2 = (pred2 == expected).mean() * 100 if pred2.shape == expected.shape else 0

        print(f"\nPrediction 1: {acc1:.1f}% {'✓ SOLVED' if match1 else ''}")
        print(f"Prediction 2: {acc2:.1f}% {'✓ SOLVED' if match2 else ''}")
    except Exception as e:
        print(f"Error testing predictions: {e}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Count hypothesis types
    diverse_types = {}
    for h in diverse_hyps:
        variant = h.parameters.get('variant', 'unknown')
        diverse_types[variant] = diverse_types.get(variant, 0) + 1

    conditional_types = {}
    for h in conditional_hyps:
        variant = h.parameters.get('variant', 'unknown')
        conditional_types[variant] = conditional_types.get(variant, 0) + 1

    print("\nDiverse solver hypothesis breakdown:")
    for variant, count in sorted(diverse_types.items()):
        print(f"  - {variant}: {count}")

    print("\nConditional solver hypothesis breakdown:")
    for variant, count in sorted(conditional_types.items()):
        print(f"  - {variant}: {count}")

    # Check if parent hypotheses are inherited
    print("\n✓ Issue identified:")
    if len(conditional_hyps) < len(diverse_hyps):
        print("  Conditional solver generated FEWER hypotheses than diverse solver")
        print("  Parent class hypotheses may not be properly inherited")
    elif 'exact' not in conditional_types and 'exact' in diverse_types:
        print("  Conditional solver missing 'exact' pattern hypotheses")
        print("  Parent class pattern detection may be broken")
    else:
        print("  Hypotheses generated but predictions are wrong")
        print("  Need to check hypothesis prioritization and selection")

if __name__ == '__main__':
    compare_hypotheses()
