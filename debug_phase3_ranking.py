"""Debug Phase 3: Check ranking and activation of nested/pipeline hypotheses"""

import json
import numpy as np
from pathlib import Path

from arc_curiosity_solver.solver_conditional import ConditionalARCCuriositySolver


def debug_ranking(solver, task_data):
    """Debug hypothesis ranking."""
    test_input = np.array(task_data['test'][0]['input'])
    train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                  for ex in task_data['train']]

    solver.verbose = False
    hypotheses = solver._generate_hypotheses(train_pairs, test_input)

    # Analyze top 10
    top_10 = hypotheses[:10] if len(hypotheses) >= 10 else hypotheses

    top_variants = []
    for h in top_10:
        variant = h.parameters.get('variant', 'other')
        activation = h.activation
        top_variants.append((variant, activation))

    # Find best nested and pipeline
    nested_hyps = [(i, h) for i, h in enumerate(hypotheses) if h.parameters.get('variant') == 'nested']
    pipeline_hyps = [(i, h) for i, h in enumerate(hypotheses) if h.parameters.get('variant') == 'pipeline']

    best_nested = None
    if nested_hyps:
        best_nested = min(nested_hyps, key=lambda x: x[0])  # Lowest index = highest rank

    best_pipeline = None
    if pipeline_hyps:
        best_pipeline = min(pipeline_hyps, key=lambda x: x[0])

    return {
        'total': len(hypotheses),
        'top_10': top_variants,
        'best_nested_rank': best_nested[0] if best_nested else None,
        'best_nested_activation': best_nested[1].activation if best_nested else None,
        'best_pipeline_rank': best_pipeline[0] if best_pipeline else None,
        'best_pipeline_activation': best_pipeline[1].activation if best_pipeline else None
    }


def main():
    print("\n" + "="*70)
    print("DEBUG: Phase 3 Hypothesis Ranking")
    print("="*70)

    training_dir = Path("ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))[:10]

    # Phase 3 solver
    phase3 = ConditionalARCCuriositySolver()
    phase3.use_nested_conditionals = True
    phase3.use_multi_stage = True

    print(f"\nAnalyzing top hypotheses for {len(task_files)} tasks...\n")

    for task_file in task_files:
        task_id = task_file.stem

        with open(task_file, 'r') as f:
            task_data = json.load(f)

        try:
            info = debug_ranking(phase3, task_data)

            print(f"{task_id} (Total: {info['total']}):")
            print(f"  Top 10 variants:")
            for i, (variant, activation) in enumerate(info['top_10']):
                print(f"    #{i+1}: {variant:15s} (activation: {activation:.3f})")

            if info['best_nested_rank'] is not None:
                print(f"  Best nested: Rank #{info['best_nested_rank']+1}, Activation: {info['best_nested_activation']:.3f}")
            else:
                print(f"  Best nested: None generated")

            if info['best_pipeline_rank'] is not None:
                print(f"  Best pipeline: Rank #{info['best_pipeline_rank']+1}, Activation: {info['best_pipeline_activation']:.3f}")
            else:
                print(f"  Best pipeline: None generated")

            print()

        except Exception as e:
            print(f"{task_id}: ERROR - {e}")

    print("="*70)
    print("KEY FINDING:")
    print("="*70)
    print("If nested/pipeline hypotheses are consistently ranked low (outside top 3),")
    print("they won't be tested even if generated. This explains why they don't")
    print("improve accuracy - the activation/confidence boosting may be insufficient.")
    print("\nPossible solutions:")
    print("  1. Increase nested_priority_boost (currently 1.8)")
    print("  2. Lower validation threshold (currently 30%)")
    print("  3. Test more hypotheses (currently top 3)")
    print("="*70)


if __name__ == '__main__':
    main()
