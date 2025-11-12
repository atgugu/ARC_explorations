"""Test Phase 6: Action Learning from Training Data

Compares two solver configurations:
1. Phase 5 (composite actions enabled, NO action learning)
2. Phase 6 (+ action learning: infer which transformations to try)
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
    print("PHASE 6 EVALUATION: ACTION LEARNING FROM TRAINING DATA")
    print("="*80)
    print("\nComparing two solver configurations:")
    print("  1. Phase 5 (composite actions enabled, NO action learning)")
    print("  2. Phase 6 (+ action learning: only try detected transformations)")
    print("\n" + "="*80)

    training_dir = Path("ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))[:30]

    # Create solver variants
    # Phase 5: Composite actions but no action learning
    phase5 = ConditionalARCCuriositySolver()
    phase5.use_composite_actions = True   # Enable Phase 5
    phase5.use_action_learning = False    # Disable action learning
    phase5.validation_threshold = 0.15    # Optimal from Phase 4

    # Phase 6: Full system with action learning
    phase6 = ConditionalARCCuriositySolver()
    phase6.use_composite_actions = True   # Enable Phase 5
    phase6.use_action_learning = True     # Enable Phase 6 action learning
    phase6.validation_threshold = 0.15    # Keep optimal threshold

    results = {
        'phase5': [],
        'phase6': []
    }

    print(f"\nTesting {len(task_files)} tasks...\n")

    for i, task_file in enumerate(task_files):
        task_id = task_file.stem

        with open(task_file, 'r') as f:
            task_data = json.load(f)

        # Test both variants
        r_p5 = test_solver(phase5, task_data)
        r_p6 = test_solver(phase6, task_data)

        r_p5['task_id'] = task_id
        r_p6['task_id'] = task_id

        results['phase5'].append(r_p5)
        results['phase6'].append(r_p6)

        # Report notable improvements
        if r_p6['solved'] and not r_p5['solved']:
            print(f"  🎉 {task_id}: NEW SOLVE by Phase 6! (Phase 5: {r_p5['accuracy']*100:.1f}%)")
        elif r_p6['accuracy'] > r_p5['accuracy'] + 0.1:
            print(f"  📈 {task_id}: Phase 5: {r_p5['accuracy']*100:.1f}% → Phase 6: {r_p6['accuracy']*100:.1f}% (+{(r_p6['accuracy']-r_p5['accuracy'])*100:.1f}%)")

        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(task_files)}")

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Solve rates
    p5_solved = sum(1 for r in results['phase5'] if r['solved'])
    p6_solved = sum(1 for r in results['phase6'] if r['solved'])

    print(f"\n🎯 EXACT SOLVE RATE:")
    print(f"  Phase 5:  {p5_solved}/{len(task_files)} ({p5_solved/len(task_files)*100:.1f}%)")
    print(f"  Phase 6:  {p6_solved}/{len(task_files)} ({p6_solved/len(task_files)*100:.1f}%)")
    print(f"  Change: {p6_solved - p5_solved:+d} ({(p6_solved - p5_solved)/len(task_files)*100:+.1f}%)")

    # Average accuracy
    p5_avg = np.mean([r['accuracy'] for r in results['phase5']]) * 100
    p6_avg = np.mean([r['accuracy'] for r in results['phase6']]) * 100

    print(f"\n📊 AVERAGE ACCURACY:")
    print(f"  Phase 5:  {p5_avg:.1f}%")
    print(f"  Phase 6:  {p6_avg:.1f}%")
    print(f"  Change: {p6_avg - p5_avg:+.1f}%")

    # Hypothesis counts
    p5_hyps = np.mean([r['hyp_count'] for r in results['phase5']])
    p6_hyps = np.mean([r['hyp_count'] for r in results['phase6']])

    print(f"\n🔍 HYPOTHESES PER TASK:")
    print(f"  Phase 5:  {p5_hyps:.1f}")
    print(f"  Phase 6:  {p6_hyps:.1f}")
    print(f"  Change: {p6_hyps - p5_hyps:+.1f} ({(p6_hyps - p5_hyps)/p5_hyps*100:+.1f}%)")

    # Hypothesis efficiency (solves per hypothesis generated)
    p5_efficiency = p5_solved / max(sum(r['hyp_count'] for r in results['phase5']), 1)
    p6_efficiency = p6_solved / max(sum(r['hyp_count'] for r in results['phase6']), 1)

    print(f"\n⚡ HYPOTHESIS EFFICIENCY (solves per 100 hypotheses):")
    print(f"  Phase 5:  {p5_efficiency * 100:.2f}")
    print(f"  Phase 6:  {p6_efficiency * 100:.2f}")
    print(f"  Change: {(p6_efficiency - p5_efficiency) * 100:+.2f}")

    # Task improvements
    p6_improved = sum(1 for i in range(len(task_files))
                      if results['phase6'][i]['accuracy'] > results['phase5'][i]['accuracy'])

    print(f"\n📈 TASKS IMPROVED (vs Phase 5):")
    print(f"  Phase 6: {p6_improved}/{len(task_files)} ({p6_improved/len(task_files)*100:.1f}%)")

    # Top improvements
    improvements = []
    for i in range(len(task_files)):
        diff = results['phase6'][i]['accuracy'] - results['phase5'][i]['accuracy']
        if diff > 0.05:
            improvements.append({
                'task': results['phase6'][i]['task_id'],
                'phase5': results['phase5'][i]['accuracy'] * 100,
                'phase6': results['phase6'][i]['accuracy'] * 100,
                'improvement': diff * 100
            })

    if improvements:
        improvements.sort(key=lambda x: x['improvement'], reverse=True)
        print(f"\n📈 TOP IMPROVEMENTS (Phase 6 over Phase 5):")
        for imp in improvements[:5]:
            print(f"  {imp['task']}: {imp['phase5']:.1f}% → {imp['phase6']:.1f}% (+{imp['improvement']:.1f}%)")

    # Newly solved
    newly_solved = []
    for i in range(len(task_files)):
        if results['phase6'][i]['solved'] and not results['phase5'][i]['solved']:
            newly_solved.append({
                'task': results['phase6'][i]['task_id'],
                'phase5': results['phase5'][i]['accuracy'] * 100
            })

    if newly_solved:
        print(f"\n🎉 NEWLY SOLVED BY PHASE 6:")
        for solve in newly_solved:
            print(f"  {solve['task']}: {solve['phase5']:.1f}% → 100% ✓")

    # Tasks with fewer hypotheses
    fewer_hyps = sum(1 for i in range(len(task_files))
                     if results['phase6'][i]['hyp_count'] < results['phase5'][i]['hyp_count'])

    print(f"\n🎯 HYPOTHESIS REDUCTION:")
    print(f"  Tasks with fewer hypotheses: {fewer_hyps}/{len(task_files)} ({fewer_hyps/len(task_files)*100:.1f}%)")

    print(f"\n{'='*80}")

    # Verdict
    gain = p6_avg - p5_avg
    hyp_reduction = (p5_hyps - p6_hyps) / p5_hyps * 100

    if p6_solved > p5_solved:
        print(f"✅ PHASE 6 SUCCESS: Breakthrough to {p6_solved - p5_solved} new exact solve(s)!")
    elif gain > 5:
        print(f"✅ PHASE 6 SUCCESS: +{gain:.1f}% accuracy gain!")
    elif gain > 2:
        print(f"📊 PHASE 6 IMPROVEMENT: +{gain:.1f}% accuracy gain")
    elif gain > 0:
        print(f"🟡 PHASE 6 MODEST IMPROVEMENT: +{gain:.1f}% accuracy gain")
    else:
        print("⚠️  PHASE 6 NO ACCURACY IMPROVEMENT")

    if hyp_reduction > 5:
        print(f"✅ HYPOTHESIS EFFICIENCY: {hyp_reduction:.1f}% fewer hypotheses generated")
    elif hyp_reduction > 0:
        print(f"🟡 HYPOTHESIS EFFICIENCY: {hyp_reduction:.1f}% fewer hypotheses generated")
    else:
        print(f"⚠️  HYPOTHESIS COUNT: {-hyp_reduction:.1f}% more hypotheses generated")

    print("="*80)

    # Summary statistics
    print(f"\n📋 SUMMARY:")
    print(f"  Phase 6 added:  {p6_avg - p5_avg:+.1f}% accuracy, {p6_solved - p5_solved:+d} solves")
    print(f"  Hypothesis change: {p6_hyps - p5_hyps:+.1f} avg per task ({hyp_reduction:+.1f}%)")
    print(f"  Key innovation: Action inference from training (detect which transformations occur)")
    print(f"  Tasks improved: {p6_improved} ({p6_improved/len(task_files)*100:.1f}%)")
    print()


if __name__ == '__main__':
    main()
