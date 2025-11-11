"""Test Phase 5: Composite Actions (Geometric & Grid Transformations)

Compares two solver configurations:
1. Phase 4 (richer predicates + threshold=0.15, no composite actions)
2. Phase 5 (+ composite actions: rotate, reflect, swap, replicate, extend)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from arc_curiosity_solver.solver_conditional import ConditionalARCCuriositySolver


def test_solver(solver, task_data):
    """Test solver on task."""
    try:
        test_input = np.array(task_data['test'][0]['input'])
        expected = np.array(task_data['test'][0]['output'])

        train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                      for ex in task_data['train']]

        solver.verbose = False
        hypotheses = solver._generate_hypotheses(train_pairs, test_input)

        if not hypotheses:
            return {'solved': False, 'accuracy': 0.0, 'hyp_count': 0}

        best_acc = 0.0
        solved = False

        for h in hypotheses[:3]:  # Test top 3
            try:
                pred = h.program.function(test_input.copy())
                if np.array_equal(pred, expected):
                    solved = True
                    best_acc = 1.0
                    break
                if pred.shape == expected.shape:
                    acc = (pred == expected).mean()
                    best_acc = max(best_acc, acc)
            except:
                pass

        return {'solved': solved, 'accuracy': best_acc, 'hyp_count': len(hypotheses)}
    except Exception as e:
        return {'solved': False, 'accuracy': 0.0, 'hyp_count': 0}


def main():
    print("\n" + "="*80)
    print("PHASE 5 EVALUATION: COMPOSITE ACTIONS (GEOMETRIC & GRID TRANSFORMATIONS)")
    print("="*80)
    print("\nComparing two solver configurations:")
    print("  1. Phase 4 (richer predicates + threshold=0.15, NO composite actions)")
    print("  2. Phase 5 (+ composite actions: rotate, reflect, swap, replicate, extend)")
    print("\n" + "="*80)

    training_dir = Path("ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))[:30]

    # Create solver variants
    # Phase 4: Best from previous phase (no composite actions)
    phase4 = ConditionalARCCuriositySolver()
    phase4.use_composite_actions = False  # Disable Phase 5
    phase4.validation_threshold = 0.15    # Optimal from Phase 4

    # Phase 5: Full system with composite actions
    phase5 = ConditionalARCCuriositySolver()
    phase5.use_composite_actions = True   # Enable Phase 5
    phase5.validation_threshold = 0.15    # Keep optimal threshold

    results = {
        'phase4': [],
        'phase5': []
    }

    print(f"\nTesting {len(task_files)} tasks...\n")

    for i, task_file in enumerate(task_files):
        task_id = task_file.stem

        with open(task_file, 'r') as f:
            task_data = json.load(f)

        # Test both variants
        r_p4 = test_solver(phase4, task_data)
        r_p5 = test_solver(phase5, task_data)

        r_p4['task_id'] = task_id
        r_p5['task_id'] = task_id

        results['phase4'].append(r_p4)
        results['phase5'].append(r_p5)

        # Report notable improvements
        if r_p5['solved'] and not r_p4['solved']:
            print(f"  🎉 {task_id}: NEW SOLVE by Phase 5! (Phase 4: {r_p4['accuracy']*100:.1f}%)")
        elif r_p5['accuracy'] > r_p4['accuracy'] + 0.1:
            print(f"  📈 {task_id}: Phase 4: {r_p4['accuracy']*100:.1f}% → Phase 5: {r_p5['accuracy']*100:.1f}% (+{(r_p5['accuracy']-r_p4['accuracy'])*100:.1f}%)")

        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(task_files)}")

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Solve rates
    p4_solved = sum(1 for r in results['phase4'] if r['solved'])
    p5_solved = sum(1 for r in results['phase5'] if r['solved'])

    print(f"\n🎯 EXACT SOLVE RATE:")
    print(f"  Phase 4:  {p4_solved}/{len(task_files)} ({p4_solved/len(task_files)*100:.1f}%)")
    print(f"  Phase 5:  {p5_solved}/{len(task_files)} ({p5_solved/len(task_files)*100:.1f}%)")
    print(f"  Change: {p5_solved - p4_solved:+d} ({(p5_solved - p4_solved)/len(task_files)*100:+.1f}%)")

    # Average accuracy
    p4_avg = np.mean([r['accuracy'] for r in results['phase4']]) * 100
    p5_avg = np.mean([r['accuracy'] for r in results['phase5']]) * 100

    print(f"\n📊 AVERAGE ACCURACY:")
    print(f"  Phase 4:  {p4_avg:.1f}%")
    print(f"  Phase 5:  {p5_avg:.1f}%")
    print(f"  Change: {p5_avg - p4_avg:+.1f}%")

    # Hypothesis counts
    p4_hyps = np.mean([r['hyp_count'] for r in results['phase4']])
    p5_hyps = np.mean([r['hyp_count'] for r in results['phase5']])

    print(f"\n🔍 HYPOTHESES PER TASK:")
    print(f"  Phase 4:  {p4_hyps:.1f}")
    print(f"  Phase 5:  {p5_hyps:.1f}")
    print(f"  Change: {p5_hyps - p4_hyps:+.1f}")

    # Task improvements
    p5_improved = sum(1 for i in range(len(task_files))
                      if results['phase5'][i]['accuracy'] > results['phase4'][i]['accuracy'])

    print(f"\n📈 TASKS IMPROVED (vs Phase 4):")
    print(f"  Phase 5: {p5_improved}/{len(task_files)} ({p5_improved/len(task_files)*100:.1f}%)")

    # Top improvements
    improvements = []
    for i in range(len(task_files)):
        diff = results['phase5'][i]['accuracy'] - results['phase4'][i]['accuracy']
        if diff > 0.05:
            improvements.append({
                'task': results['phase5'][i]['task_id'],
                'phase4': results['phase4'][i]['accuracy'] * 100,
                'phase5': results['phase5'][i]['accuracy'] * 100,
                'improvement': diff * 100
            })

    if improvements:
        improvements.sort(key=lambda x: x['improvement'], reverse=True)
        print(f"\n📈 TOP IMPROVEMENTS (Phase 5 over Phase 4):")
        for imp in improvements[:5]:
            print(f"  {imp['task']}: {imp['phase4']:.1f}% → {imp['phase5']:.1f}% (+{imp['improvement']:.1f}%)")

    # Newly solved
    newly_solved = []
    for i in range(len(task_files)):
        if results['phase5'][i]['solved'] and not results['phase4'][i]['solved']:
            newly_solved.append({
                'task': results['phase5'][i]['task_id'],
                'phase4': results['phase4'][i]['accuracy'] * 100
            })

    if newly_solved:
        print(f"\n🎉 NEWLY SOLVED BY PHASE 5:")
        for solve in newly_solved:
            print(f"  {solve['task']}: {solve['phase4']:.1f}% → 100% ✓")

    print(f"\n{'='*80}")

    # Verdict
    gain = p5_avg - p4_avg
    if p5_solved > p4_solved:
        print(f"✅ PHASE 5 SUCCESS: Breakthrough to {p5_solved - p4_solved} new exact solve(s)!")
    elif gain > 5:
        print(f"✅ PHASE 5 SUCCESS: +{gain:.1f}% accuracy gain!")
    elif gain > 2:
        print(f"📊 PHASE 5 IMPROVEMENT: +{gain:.1f}% accuracy gain")
    elif gain > 0:
        print(f"🟡 PHASE 5 MODEST IMPROVEMENT: +{gain:.1f}% accuracy gain")
    else:
        print("⚠️  PHASE 5 NO IMPROVEMENT: Composite actions not helping yet")

    print("="*80)

    # Summary statistics
    print(f"\n📋 SUMMARY:")
    print(f"  Phase 5 added:  {p5_avg - p4_avg:+.1f}% accuracy, {p5_solved - p4_solved:+d} solves")
    print(f"  New actions tested: rotate (3 angles), reflect (2 axes), swap, replicate, extend")
    print(f"  Composite conditions: {p5_improved} tasks improved ({p5_improved/len(task_files)*100:.1f}%)")
    print()


if __name__ == '__main__':
    main()
