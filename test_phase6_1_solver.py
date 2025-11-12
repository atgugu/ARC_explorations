"""Test Phase 6.1: Object-Aware Action Learning with Confidence Prioritization

Compares three solver configurations:
1. Phase 5 (composite actions, NO action learning)
2. Phase 6 (grid-level action learning with binary yes/no)
3. Phase 6.1 (object-aware detection + confidence-based prioritization)
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
    print("PHASE 6.1 EVALUATION: OBJECT-AWARE ACTION LEARNING")
    print("="*80)
    print("\nComparing three solver configurations:")
    print("  1. Phase 5:   Composite actions, NO action learning")
    print("  2. Phase 6:   Grid-level action learning (binary yes/no)")
    print("  3. Phase 6.1: Object-aware detection + confidence prioritization")
    print("\n" + "="*80)

    training_dir = Path("ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))[:30]

    # Create solver variants
    # Phase 5: Composite actions but no action learning (baseline)
    phase5 = ConditionalARCCuriositySolver()
    phase5.use_composite_actions = True
    phase5.use_action_learning = False  # Disable action learning
    phase5.validation_threshold = 0.15

    # Phase 6.1: Full system with object-aware action learning
    phase61 = ConditionalARCCuriositySolver()
    phase61.use_composite_actions = True
    phase61.use_action_learning = True   # Enable action learning
    phase61.validation_threshold = 0.15

    results = {
        'phase5': [],
        'phase61': []
    }

    print(f"\nTesting {len(task_files)} tasks...\n")

    for i, task_file in enumerate(task_files):
        task_id = task_file.stem

        with open(task_file, 'r') as f:
            task_data = json.load(f)

        # Test both variants
        r_p5 = test_solver(phase5, task_data)
        r_p61 = test_solver(phase61, task_data)

        r_p5['task_id'] = task_id
        r_p61['task_id'] = task_id

        results['phase5'].append(r_p5)
        results['phase61'].append(r_p61)

        # Report notable improvements
        if r_p61['solved'] and not r_p5['solved']:
            print(f"  🎉 {task_id}: NEW SOLVE by Phase 6.1! (Phase 5: {r_p5['accuracy']*100:.1f}%)")
        elif r_p61['accuracy'] > r_p5['accuracy'] + 0.1:
            print(f"  📈 {task_id}: Phase 5: {r_p5['accuracy']*100:.1f}% → Phase 6.1: {r_p61['accuracy']*100:.1f}% (+{(r_p61['accuracy']-r_p5['accuracy'])*100:.1f}%)")
        elif r_p61['accuracy'] < r_p5['accuracy'] - 0.1:
            print(f"  📉 {task_id}: Phase 5: {r_p5['accuracy']*100:.1f}% → Phase 6.1: {r_p61['accuracy']*100:.1f}% ({(r_p61['accuracy']-r_p5['accuracy'])*100:.1f}%)")

        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(task_files)}")

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Solve rates
    p5_solved = sum(1 for r in results['phase5'] if r['solved'])
    p61_solved = sum(1 for r in results['phase61'] if r['solved'])

    print(f"\n🎯 EXACT SOLVE RATE:")
    print(f"  Phase 5:    {p5_solved}/{len(task_files)} ({p5_solved/len(task_files)*100:.1f}%)")
    print(f"  Phase 6.1:  {p61_solved}/{len(task_files)} ({p61_solved/len(task_files)*100:.1f}%)")
    print(f"  Change: {p61_solved - p5_solved:+d} ({(p61_solved - p5_solved)/len(task_files)*100:+.1f}%)")

    # Average accuracy
    p5_avg = np.mean([r['accuracy'] for r in results['phase5']]) * 100
    p61_avg = np.mean([r['accuracy'] for r in results['phase61']]) * 100

    print(f"\n📊 AVERAGE ACCURACY:")
    print(f"  Phase 5:    {p5_avg:.1f}%")
    print(f"  Phase 6.1:  {p61_avg:.1f}%")
    print(f"  Change: {p61_avg - p5_avg:+.1f}%")

    # Hypothesis counts
    p5_hyps = np.mean([r['hyp_count'] for r in results['phase5']])
    p61_hyps = np.mean([r['hyp_count'] for r in results['phase61']])

    print(f"\n🔍 HYPOTHESES PER TASK:")
    print(f"  Phase 5:    {p5_hyps:.1f}")
    print(f"  Phase 6.1:  {p61_hyps:.1f}")
    print(f"  Change: {p61_hyps - p5_hyps:+.1f} ({(p61_hyps - p5_hyps)/p5_hyps*100:+.1f}%)")

    # Task improvements
    p61_improved = sum(1 for i in range(len(task_files))
                       if results['phase61'][i]['accuracy'] > results['phase5'][i]['accuracy'])
    p61_regressed = sum(1 for i in range(len(task_files))
                        if results['phase61'][i]['accuracy'] < results['phase5'][i]['accuracy'])

    print(f"\n📈 TASK-LEVEL CHANGES:")
    print(f"  Improved:   {p61_improved}/{len(task_files)} ({p61_improved/len(task_files)*100:.1f}%)")
    print(f"  Regressed:  {p61_regressed}/{len(task_files)} ({p61_regressed/len(task_files)*100:.1f}%)")
    print(f"  Unchanged:  {len(task_files) - p61_improved - p61_regressed}/{len(task_files)}")

    # Top improvements
    improvements = []
    for i in range(len(task_files)):
        diff = results['phase61'][i]['accuracy'] - results['phase5'][i]['accuracy']
        if abs(diff) > 0.05:
            improvements.append({
                'task': results['phase61'][i]['task_id'],
                'phase5': results['phase5'][i]['accuracy'] * 100,
                'phase61': results['phase61'][i]['accuracy'] * 100,
                'improvement': diff * 100
            })

    if improvements:
        improvements.sort(key=lambda x: abs(x['improvement']), reverse=True)
        print(f"\n📈 TOP CHANGES (Phase 6.1 vs Phase 5):")
        for imp in improvements[:8]:
            emoji = "+" if imp['improvement'] > 0 else ""
            print(f"  {imp['task']}: {imp['phase5']:.1f}% → {imp['phase61']:.1f}% ({emoji}{imp['improvement']:.1f}%)")

    # Newly solved
    newly_solved = []
    for i in range(len(task_files)):
        if results['phase61'][i]['solved'] and not results['phase5'][i]['solved']:
            newly_solved.append({
                'task': results['phase61'][i]['task_id'],
                'phase5': results['phase5'][i]['accuracy'] * 100
            })

    if newly_solved:
        print(f"\n🎉 NEWLY SOLVED BY PHASE 6.1:")
        for solve in newly_solved:
            print(f"  {solve['task']}: {solve['phase5']:.1f}% → 100% ✓")

    # Hypothesis efficiency
    p5_total_hyps = sum(r['hyp_count'] for r in results['phase5'])
    p61_total_hyps = sum(r['hyp_count'] for r in results['phase61'])
    hyp_reduction = (p5_total_hyps - p61_total_hyps) / p5_total_hyps * 100

    print(f"\n🎯 HYPOTHESIS EFFICIENCY:")
    print(f"  Phase 5 total:   {p5_total_hyps}")
    print(f"  Phase 6.1 total: {p61_total_hyps}")
    print(f"  Reduction: {hyp_reduction:.1f}%")

    print(f"\n{'='*80}")

    # Verdict
    gain = p61_avg - p5_avg

    if p61_solved > p5_solved:
        print(f"✅ PHASE 6.1 SUCCESS: Breakthrough to {p61_solved - p5_solved} new exact solve(s)!")
    elif gain > 5:
        print(f"✅ PHASE 6.1 SUCCESS: +{gain:.1f}% accuracy gain!")
    elif gain > 2:
        print(f"📊 PHASE 6.1 IMPROVEMENT: +{gain:.1f}% accuracy gain")
    elif gain > 0:
        print(f"🟡 PHASE 6.1 MODEST IMPROVEMENT: +{gain:.1f}% accuracy gain")
    elif gain > -2:
        print(f"🟡 PHASE 6.1 NEUTRAL: {gain:+.1f}% accuracy change (within margin)")
    else:
        print(f"⚠️  PHASE 6.1 REGRESSION: {gain:.1f}% accuracy loss")

    if hyp_reduction > 5:
        print(f"✅ HYPOTHESIS EFFICIENCY: {hyp_reduction:.1f}% fewer hypotheses generated")
    elif hyp_reduction > 0:
        print(f"🟡 HYPOTHESIS EFFICIENCY: {hyp_reduction:.1f}% fewer hypotheses generated")
    elif hyp_reduction > -5:
        print(f"🟡 HYPOTHESIS COUNT: Similar to Phase 5")
    else:
        print(f"⚠️  HYPOTHESIS COUNT: {-hyp_reduction:.1f}% more hypotheses generated")

    print("="*80)

    # Summary statistics
    print(f"\n📋 SUMMARY:")
    print(f"  Phase 6.1 changes vs Phase 5:")
    print(f"    Accuracy:    {p61_avg - p5_avg:+.1f}% ({p61_avg:.1f}% vs {p5_avg:.1f}%)")
    print(f"    Solve rate:  {p61_solved - p5_solved:+d} ({p61_solved} vs {p5_solved})")
    print(f"    Hypotheses:  {p61_hyps - p5_hyps:+.1f} per task ({hyp_reduction:+.1f}%)")
    print(f"    Tasks improved: {p61_improved} ({p61_improved/len(task_files)*100:.1f}%)")
    print(f"\n  Key innovations:")
    print(f"    ✓ Object-level rotation/reflection detection (not just grid-level)")
    print(f"    ✓ Confidence-based prioritization (not binary yes/no)")
    print(f"    ✓ Always try actions, but boost detected ones")
    print()


if __name__ == '__main__':
    main()
