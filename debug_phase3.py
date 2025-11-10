"""Debug Phase 3: Check if nested/pipeline hypotheses are being generated"""

import json
import numpy as np
from pathlib import Path

from arc_curiosity_solver.solver_conditional import ConditionalARCCuriositySolver


def debug_hypotheses(solver, task_data):
    """Debug hypothesis generation."""
    test_input = np.array(task_data['test'][0]['input'])
    train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                  for ex in task_data['train']]

    solver.verbose = False
    hypotheses = solver._generate_hypotheses(train_pairs, test_input)

    # Count by variant type
    counts = {
        'exact': 0,
        'variation': 0,
        'spatial': 0,
        'conditional': 0,
        'nested': 0,
        'pipeline': 0,
        'other': 0
    }

    for h in hypotheses:
        variant = h.parameters.get('variant', 'other')
        if variant in counts:
            counts[variant] += 1
        else:
            counts['other'] += 1

    return counts, len(hypotheses)


def main():
    print("\n" + "="*70)
    print("DEBUG: Phase 3 Hypothesis Generation")
    print("="*70)

    training_dir = Path("ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))[:10]  # Just 10 tasks

    # Phase 3 solver
    phase3 = ConditionalARCCuriositySolver()
    phase3.use_nested_conditionals = True
    phase3.use_multi_stage = True

    print(f"\nTesting {len(task_files)} tasks to check hypothesis types...\n")

    total_counts = {
        'exact': 0,
        'variation': 0,
        'spatial': 0,
        'conditional': 0,
        'nested': 0,
        'pipeline': 0,
        'other': 0
    }

    for task_file in task_files:
        task_id = task_file.stem

        with open(task_file, 'r') as f:
            task_data = json.load(f)

        try:
            counts, total = debug_hypotheses(phase3, task_data)

            print(f"{task_id}:")
            print(f"  Total: {total}")
            print(f"  Exact: {counts['exact']}, Variations: {counts['variation']}, Spatial: {counts['spatial']}")
            print(f"  Conditional: {counts['conditional']}, Nested: {counts['nested']}, Pipeline: {counts['pipeline']}")
            print()

            for key in total_counts:
                total_counts[key] += counts[key]

        except Exception as e:
            print(f"{task_id}: ERROR - {e}")

    print("="*70)
    print("TOTALS ACROSS ALL TASKS:")
    print("="*70)
    print(f"  Exact:       {total_counts['exact']}")
    print(f"  Variations:  {total_counts['variation']}")
    print(f"  Spatial:     {total_counts['spatial']}")
    print(f"  Conditional: {total_counts['conditional']}")
    print(f"  Nested:      {total_counts['nested']}")
    print(f"  Pipeline:    {total_counts['pipeline']}")
    print(f"  Other:       {total_counts['other']}")
    print(f"\n  Total:       {sum(total_counts.values())}")

    if total_counts['nested'] == 0:
        print("\n⚠️  NO NESTED CONDITIONALS GENERATED!")
        print("   Possible reasons:")
        print("   - Validation threshold too strict (30%)")
        print("   - Property combinations not matching training patterns")
        print("   - Implementation bug in _generate_nested_conditionals()")

    if total_counts['pipeline'] == 0:
        print("\n⚠️  NO MULTI-STAGE PIPELINES GENERATED!")
        print("   Possible reasons:")
        print("   - Validation threshold too strict (30%)")
        print("   - Sequential patterns not matching training")
        print("   - Implementation bug in _generate_multi_stage_pipelines()")

    print("="*70)


if __name__ == '__main__':
    main()
