"""Test Phase 4: Richer Predicates + Validation Threshold Tuning

Compares multiple solver configurations:
1. Phase 3 (baseline with basic predicates)
2. Phase 4 with threshold = 0.30 (strictest)
3. Phase 4 with threshold = 0.25
4. Phase 4 with threshold = 0.20
5. Phase 4 with threshold = 0.15 (most permissive)
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
    print("PHASE 4 EVALUATION: RICHER PREDICATES + VALIDATION THRESHOLD TUNING")
    print("="*80)
    print("\nComparing five solver configurations:")
    print("  1. Phase 3 (baseline - no richer predicates)")
    print("  2. Phase 4 threshold=0.30 (strictest)")
    print("  3. Phase 4 threshold=0.25")
    print("  4. Phase 4 threshold=0.20")
    print("  5. Phase 4 threshold=0.15 (most permissive)")
    print("\n" + "="*80)

    training_dir = Path("ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))[:30]

    # Create solver variants
    # Phase 3: Baseline (no richer predicates)
    phase3 = ConditionalARCCuriositySolver()
    phase3.use_richer_predicates = False
    phase3.validation_threshold = 0.30

    # Phase 4 with different thresholds
    phase4_30 = ConditionalARCCuriositySolver()
    phase4_30.use_richer_predicates = True
    phase4_30.validation_threshold = 0.30

    phase4_25 = ConditionalARCCuriositySolver()
    phase4_25.use_richer_predicates = True
    phase4_25.validation_threshold = 0.25

    phase4_20 = ConditionalARCCuriositySolver()
    phase4_20.use_richer_predicates = True
    phase4_20.validation_threshold = 0.20

    phase4_15 = ConditionalARCCuriositySolver()
    phase4_15.use_richer_predicates = True
    phase4_15.validation_threshold = 0.15

    results = {
        'phase3': [],
        'p4_30': [],
        'p4_25': [],
        'p4_20': [],
        'p4_15': []
    }

    print(f"\nTesting {len(task_files)} tasks...\n")

    for i, task_file in enumerate(task_files):
        task_id = task_file.stem

        with open(task_file, 'r') as f:
            task_data = json.load(f)

        # Test all five variants
        r_p3 = test_solver(phase3, task_data)
        r_30 = test_solver(phase4_30, task_data)
        r_25 = test_solver(phase4_25, task_data)
        r_20 = test_solver(phase4_20, task_data)
        r_15 = test_solver(phase4_15, task_data)

        r_p3['task_id'] = task_id
        r_30['task_id'] = task_id
        r_25['task_id'] = task_id
        r_20['task_id'] = task_id
        r_15['task_id'] = task_id

        results['phase3'].append(r_p3)
        results['p4_30'].append(r_30)
        results['p4_25'].append(r_25)
        results['p4_20'].append(r_20)
        results['p4_15'].append(r_15)

        # Report notable improvements
        best_p4 = max(r_30['accuracy'], r_25['accuracy'], r_20['accuracy'], r_15['accuracy'])
        if best_p4 > r_p3['accuracy'] + 0.1:
            print(f"  📈 {task_id}: Phase 3: {r_p3['accuracy']*100:.1f}% → Best Phase 4: {best_p4*100:.1f}% (+{(best_p4-r_p3['accuracy'])*100:.1f}%)")

        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(task_files)}")

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Solve rates
    p3_solved = sum(1 for r in results['phase3'] if r['solved'])
    p4_30_solved = sum(1 for r in results['p4_30'] if r['solved'])
    p4_25_solved = sum(1 for r in results['p4_25'] if r['solved'])
    p4_20_solved = sum(1 for r in results['p4_20'] if r['solved'])
    p4_15_solved = sum(1 for r in results['p4_15'] if r['solved'])

    print(f"\n🎯 EXACT SOLVE RATE:")
    print(f"  Phase 3:        {p3_solved}/{len(task_files)} ({p3_solved/len(task_files)*100:.1f}%)")
    print(f"  Phase 4 (t=0.30): {p4_30_solved}/{len(task_files)} ({p4_30_solved/len(task_files)*100:.1f}%) [{p4_30_solved-p3_solved:+d}]")
    print(f"  Phase 4 (t=0.25): {p4_25_solved}/{len(task_files)} ({p4_25_solved/len(task_files)*100:.1f}%) [{p4_25_solved-p3_solved:+d}]")
    print(f"  Phase 4 (t=0.20): {p4_20_solved}/{len(task_files)} ({p4_20_solved/len(task_files)*100:.1f}%) [{p4_20_solved-p3_solved:+d}]")
    print(f"  Phase 4 (t=0.15): {p4_15_solved}/{len(task_files)} ({p4_15_solved/len(task_files)*100:.1f}%) [{p4_15_solved-p3_solved:+d}]")

    # Average accuracy
    p3_avg = np.mean([r['accuracy'] for r in results['phase3']]) * 100
    p4_30_avg = np.mean([r['accuracy'] for r in results['p4_30']]) * 100
    p4_25_avg = np.mean([r['accuracy'] for r in results['p4_25']]) * 100
    p4_20_avg = np.mean([r['accuracy'] for r in results['p4_20']]) * 100
    p4_15_avg = np.mean([r['accuracy'] for r in results['p4_15']]) * 100

    print(f"\n📊 AVERAGE ACCURACY:")
    print(f"  Phase 3:        {p3_avg:.1f}%")
    print(f"  Phase 4 (t=0.30): {p4_30_avg:.1f}% [{p4_30_avg-p3_avg:+.1f}%]")
    print(f"  Phase 4 (t=0.25): {p4_25_avg:.1f}% [{p4_25_avg-p3_avg:+.1f}%]")
    print(f"  Phase 4 (t=0.20): {p4_20_avg:.1f}% [{p4_20_avg-p3_avg:+.1f}%]")
    print(f"  Phase 4 (t=0.15): {p4_15_avg:.1f}% [{p4_15_avg-p3_avg:+.1f}%]")

    # Find best threshold
    best_threshold = None
    best_avg = p3_avg
    if p4_30_avg > best_avg:
        best_threshold = "0.30"
        best_avg = p4_30_avg
    if p4_25_avg > best_avg:
        best_threshold = "0.25"
        best_avg = p4_25_avg
    if p4_20_avg > best_avg:
        best_threshold = "0.20"
        best_avg = p4_20_avg
    if p4_15_avg > best_avg:
        best_threshold = "0.15"
        best_avg = p4_15_avg

    if best_threshold:
        print(f"\n🏆 BEST THRESHOLD: {best_threshold} ({best_avg:.1f}% accuracy, +{best_avg-p3_avg:.1f}%)")
    else:
        print(f"\n⚠️  NO IMPROVEMENT: Phase 3 still best ({p3_avg:.1f}% accuracy)")

    # Hypothesis counts
    p3_hyps = np.mean([r['hyp_count'] for r in results['phase3']])
    p4_30_hyps = np.mean([r['hyp_count'] for r in results['p4_30']])
    p4_25_hyps = np.mean([r['hyp_count'] for r in results['p4_25']])
    p4_20_hyps = np.mean([r['hyp_count'] for r in results['p4_20']])
    p4_15_hyps = np.mean([r['hyp_count'] for r in results['p4_15']])

    print(f"\n🔍 HYPOTHESES PER TASK:")
    print(f"  Phase 3:        {p3_hyps:.1f}")
    print(f"  Phase 4 (t=0.30): {p4_30_hyps:.1f} [{p4_30_hyps-p3_hyps:+.1f}]")
    print(f"  Phase 4 (t=0.25): {p4_25_hyps:.1f} [{p4_25_hyps-p3_hyps:+.1f}]")
    print(f"  Phase 4 (t=0.20): {p4_20_hyps:.1f} [{p4_20_hyps-p3_hyps:+.1f}]")
    print(f"  Phase 4 (t=0.15): {p4_15_hyps:.1f} [{p4_15_hyps-p3_hyps:+.1f}]")

    # Task improvements
    p4_30_improved = sum(1 for i in range(len(task_files))
                         if results['p4_30'][i]['accuracy'] > results['phase3'][i]['accuracy'])
    p4_25_improved = sum(1 for i in range(len(task_files))
                         if results['p4_25'][i]['accuracy'] > results['phase3'][i]['accuracy'])
    p4_20_improved = sum(1 for i in range(len(task_files))
                         if results['p4_20'][i]['accuracy'] > results['phase3'][i]['accuracy'])
    p4_15_improved = sum(1 for i in range(len(task_files))
                         if results['p4_15'][i]['accuracy'] > results['phase3'][i]['accuracy'])

    print(f"\n📈 TASKS IMPROVED (vs Phase 3):")
    print(f"  Phase 4 (t=0.30): {p4_30_improved}/{len(task_files)} ({p4_30_improved/len(task_files)*100:.1f}%)")
    print(f"  Phase 4 (t=0.25): {p4_25_improved}/{len(task_files)} ({p4_25_improved/len(task_files)*100:.1f}%)")
    print(f"  Phase 4 (t=0.20): {p4_20_improved}/{len(task_files)} ({p4_20_improved/len(task_files)*100:.1f}%)")
    print(f"  Phase 4 (t=0.15): {p4_15_improved}/{len(task_files)} ({p4_15_improved/len(task_files)*100:.1f}%)")

    # Top improvements for best threshold
    if best_threshold:
        # Convert "0.15" → "p4_15", "0.30" → "p4_30"
        threshold_key = f"p4_{best_threshold.replace('0.', '')}"
        improvements = []
        for i in range(len(task_files)):
            diff = results[threshold_key][i]['accuracy'] - results['phase3'][i]['accuracy']
            if diff > 0.05:
                improvements.append({
                    'task': results[threshold_key][i]['task_id'],
                    'phase3': results['phase3'][i]['accuracy'] * 100,
                    'phase4': results[threshold_key][i]['accuracy'] * 100,
                    'improvement': diff * 100
                })

        if improvements:
            improvements.sort(key=lambda x: x['improvement'], reverse=True)
            print(f"\n📈 TOP IMPROVEMENTS (threshold={best_threshold}):")
            for imp in improvements[:5]:
                print(f"  {imp['task']}: {imp['phase3']:.1f}% → {imp['phase4']:.1f}% (+{imp['improvement']:.1f}%)")

    print(f"\n{'='*80}")

    # Verdict
    if best_threshold:
        gain = best_avg - p3_avg
        if gain > 5:
            print(f"✅ PHASE 4 SUCCESS: +{gain:.1f}% accuracy gain!")
        elif gain > 2:
            print(f"📊 PHASE 4 IMPROVEMENT: +{gain:.1f}% accuracy gain")
        else:
            print(f"🟡 PHASE 4 MODEST IMPROVEMENT: +{gain:.1f}% accuracy gain")
    else:
        print("⚠️  PHASE 4 NO IMPROVEMENT: Richer predicates not helping yet")

    print("="*80)

    # Threshold analysis
    print(f"\n📋 THRESHOLD ANALYSIS:")
    print(f"  Lower threshold → More hypotheses → More coverage BUT more noise")
    print(f"  Higher threshold → Fewer hypotheses → Less noise BUT less coverage")
    print(f"\n  Results:")
    print(f"    0.30 (strictest):     {p4_30_avg:.1f}% avg, {p4_30_hyps:.0f} hyps/task")
    print(f"    0.25:                 {p4_25_avg:.1f}% avg, {p4_25_hyps:.0f} hyps/task")
    print(f"    0.20:                 {p4_20_avg:.1f}% avg, {p4_20_hyps:.0f} hyps/task")
    print(f"    0.15 (most permissive): {p4_15_avg:.1f}% avg, {p4_15_hyps:.0f} hyps/task")
    if best_threshold:
        print(f"\n  ✅ Optimal threshold: {best_threshold}")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
