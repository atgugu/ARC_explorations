"""Test Phase 3: Nested Conditionals + Multi-Stage Pipelines

Compares three solver variants:
1. Diverse (baseline)
2. Phase 2 (improved conditional with validation)
3. Phase 3 (+ nested conditionals + multi-stage pipelines)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from arc_curiosity_solver.solver_diverse import DiverseARCCuriositySolver
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
    print("PHASE 3 EVALUATION: NESTED CONDITIONALS + MULTI-STAGE PIPELINES")
    print("="*80)
    print("\nComparing three solver variants:")
    print("  1. Diverse (baseline)")
    print("  2. Phase 2 (improved conditional with validation)")
    print("  3. Phase 3 (+ nested conditionals + multi-stage pipelines)")
    print("\n" + "="*80)

    training_dir = Path("ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))[:30]

    # Create solver variants
    diverse = DiverseARCCuriositySolver()

    # Phase 2: Improved conditional with validation
    phase2 = ConditionalARCCuriositySolver()
    phase2.use_nested_conditionals = False
    phase2.use_multi_stage = False

    # Phase 3: Full system with nested + multi-stage
    phase3 = ConditionalARCCuriositySolver()
    phase3.use_nested_conditionals = True
    phase3.use_multi_stage = True

    results = {
        'diverse': [],
        'phase2': [],
        'phase3': []
    }

    print(f"\nTesting {len(task_files)} tasks...\n")

    for i, task_file in enumerate(task_files):
        task_id = task_file.stem

        with open(task_file, 'r') as f:
            task_data = json.load(f)

        # Test all three variants
        r_div = test_solver(diverse, task_data)
        r_p2 = test_solver(phase2, task_data)
        r_p3 = test_solver(phase3, task_data)

        r_div['task_id'] = task_id
        r_p2['task_id'] = task_id
        r_p3['task_id'] = task_id

        results['diverse'].append(r_div)
        results['phase2'].append(r_p2)
        results['phase3'].append(r_p3)

        # Report notable improvements
        if r_p3['solved'] and not r_p2['solved']:
            print(f"  🎉 {task_id}: NEW SOLVE by Phase 3! (Phase 2: {r_p2['accuracy']*100:.1f}%)")
        elif r_p3['accuracy'] > r_p2['accuracy'] + 0.1:
            print(f"  📈 {task_id}: Phase 2: {r_p2['accuracy']*100:.1f}% → Phase 3: {r_p3['accuracy']*100:.1f}% (+{(r_p3['accuracy']-r_p2['accuracy'])*100:.1f}%)")

        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(task_files)}")

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Solve rates
    div_solved = sum(1 for r in results['diverse'] if r['solved'])
    p2_solved = sum(1 for r in results['phase2'] if r['solved'])
    p3_solved = sum(1 for r in results['phase3'] if r['solved'])

    div_rate = div_solved / len(task_files) * 100
    p2_rate = p2_solved / len(task_files) * 100
    p3_rate = p3_solved / len(task_files) * 100

    print(f"\n🎯 EXACT SOLVE RATE:")
    print(f"  Diverse:  {div_solved}/{len(task_files)} ({div_rate:.1f}%)")
    print(f"  Phase 2:  {p2_solved}/{len(task_files)} ({p2_rate:.1f}%)")
    print(f"  Phase 3:  {p3_solved}/{len(task_files)} ({p3_rate:.1f}%)")
    print(f"  Phase 2 vs Baseline: {p2_solved - div_solved:+d} ({p2_rate - div_rate:+.1f}%)")
    print(f"  Phase 3 vs Phase 2:  {p3_solved - p2_solved:+d} ({p3_rate - p2_rate:+.1f}%)")
    print(f"  Phase 3 vs Baseline: {p3_solved - div_solved:+d} ({p3_rate - div_rate:+.1f}%)")

    # Average accuracy
    div_avg = np.mean([r['accuracy'] for r in results['diverse']]) * 100
    p2_avg = np.mean([r['accuracy'] for r in results['phase2']]) * 100
    p3_avg = np.mean([r['accuracy'] for r in results['phase3']]) * 100

    print(f"\n📊 AVERAGE ACCURACY:")
    print(f"  Diverse:  {div_avg:.1f}%")
    print(f"  Phase 2:  {p2_avg:.1f}%")
    print(f"  Phase 3:  {p3_avg:.1f}%")
    print(f"  Phase 2 vs Baseline: {p2_avg - div_avg:+.1f}%")
    print(f"  Phase 3 vs Phase 2:  {p3_avg - p2_avg:+.1f}%")
    print(f"  Phase 3 vs Baseline: {p3_avg - div_avg:+.1f}%")

    # Hypothesis counts
    div_hyps = np.mean([r['hyp_count'] for r in results['diverse']])
    p2_hyps = np.mean([r['hyp_count'] for r in results['phase2']])
    p3_hyps = np.mean([r['hyp_count'] for r in results['phase3']])

    print(f"\n🔍 HYPOTHESES PER TASK:")
    print(f"  Diverse:  {div_hyps:.1f}")
    print(f"  Phase 2:  {p2_hyps:.1f}")
    print(f"  Phase 3:  {p3_hyps:.1f}")

    # Task improvements
    p2_improved = sum(1 for i in range(len(task_files))
                      if results['phase2'][i]['accuracy'] > results['diverse'][i]['accuracy'])
    p3_improved = sum(1 for i in range(len(task_files))
                      if results['phase3'][i]['accuracy'] > results['phase2'][i]['accuracy'])
    p3_total_improved = sum(1 for i in range(len(task_files))
                            if results['phase3'][i]['accuracy'] > results['diverse'][i]['accuracy'])

    print(f"\n📈 TASKS IMPROVED:")
    print(f"  Phase 2 vs Baseline: {p2_improved}/{len(task_files)} ({p2_improved/len(task_files)*100:.1f}%)")
    print(f"  Phase 3 vs Phase 2:  {p3_improved}/{len(task_files)} ({p3_improved/len(task_files)*100:.1f}%)")
    print(f"  Phase 3 vs Baseline: {p3_total_improved}/{len(task_files)} ({p3_total_improved/len(task_files)*100:.1f}%)")

    # Top Phase 3 improvements over Phase 2
    improvements = []
    for i in range(len(task_files)):
        diff = results['phase3'][i]['accuracy'] - results['phase2'][i]['accuracy']
        if diff > 0.05:
            improvements.append({
                'task': results['phase3'][i]['task_id'],
                'phase2': results['phase2'][i]['accuracy'] * 100,
                'phase3': results['phase3'][i]['accuracy'] * 100,
                'improvement': diff * 100
            })

    if improvements:
        improvements.sort(key=lambda x: x['improvement'], reverse=True)
        print(f"\n📈 TOP PHASE 3 IMPROVEMENTS (over Phase 2):")
        for imp in improvements[:5]:
            print(f"  {imp['task']}: {imp['phase2']:.1f}% → {imp['phase3']:.1f}% (+{imp['improvement']:.1f}%)")

    # Newly solved by Phase 3
    newly_solved_p3 = []
    for i in range(len(task_files)):
        if results['phase3'][i]['solved'] and not results['phase2'][i]['solved']:
            newly_solved_p3.append({
                'task': results['phase3'][i]['task_id'],
                'phase2': results['phase2'][i]['accuracy'] * 100
            })

    if newly_solved_p3:
        print(f"\n🎉 NEWLY SOLVED BY PHASE 3:")
        for solve in newly_solved_p3:
            print(f"  {solve['task']}: {solve['phase2']:.1f}% → 100% ✓")

    print(f"\n{'='*80}")

    # Verdict
    if p3_solved > p2_solved:
        print("✅ PHASE 3 SUCCESS: Breakthrough to new exact solves!")
    elif p3_avg > p2_avg + 2:
        print("📊 PHASE 3 IMPROVEMENT: Significant accuracy gain")
    elif p3_avg > p2_avg:
        print("🟡 PHASE 3 MODEST IMPROVEMENT: Small accuracy gain")
    else:
        print("⚠️  PHASE 3 NO IMPROVEMENT: Nested/multi-stage not helping yet")

    print("="*80)

    # Summary statistics
    print(f"\n📋 SUMMARY:")
    print(f"  Phase 2 gained: {p2_avg - div_avg:+.1f}% accuracy, {p2_solved - div_solved:+d} solves")
    print(f"  Phase 3 added:  {p3_avg - p2_avg:+.1f}% accuracy, {p3_solved - p2_solved:+d} solves")
    print(f"  Total gain:     {p3_avg - div_avg:+.1f}% accuracy, {p3_solved - div_solved:+d} solves")
    print()


if __name__ == '__main__':
    main()
