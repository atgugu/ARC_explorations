"""Test improved conditional solver with validation and training-specific patterns"""

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
    print("\n" + "="*70)
    print("IMPROVED CONDITIONAL SOLVER TEST")
    print("="*70)
    print("\nTesting 30 tasks with improvements:")
    print("  ✅ Training-specific colors (not all 0-9)")
    print("  ✅ Validation on training data")
    print("  ✅ Relaxed object matching")
    print("  ✅ Richer conditional types\n")

    training_dir = Path("ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))[:30]

    diverse = DiverseARCCuriositySolver()
    improved = ConditionalARCCuriositySolver()

    div_results = []
    imp_results = []

    div_solved = []
    imp_solved = []
    newly_solved = []

    print(f"Testing {len(task_files)} tasks...\n")

    for i, task_file in enumerate(task_files):
        task_id = task_file.stem

        with open(task_file, 'r') as f:
            task_data = json.load(f)

        # Test both
        r_div = test_solver(diverse, task_data)
        r_imp = test_solver(improved, task_data)

        r_div['task_id'] = task_id
        r_imp['task_id'] = task_id

        div_results.append(r_div)
        imp_results.append(r_imp)

        if r_div['solved']:
            div_solved.append(task_id)
        if r_imp['solved']:
            imp_solved.append(task_id)
            if task_id not in div_solved:
                newly_solved.append(task_id)
                print(f"  🎉 {task_id}: NEW SOLVE by improved solver!")

        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(task_files)}")

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    # Solve rates
    div_rate = len(div_solved) / len(task_files) * 100
    imp_rate = len(imp_solved) / len(task_files) * 100

    print(f"\n🎯 EXACT SOLVE RATE:")
    print(f"  Diverse:  {len(div_solved)}/{len(task_files)} ({div_rate:.1f}%)")
    print(f"  Improved: {len(imp_solved)}/{len(task_files)} ({imp_rate:.1f}%)")
    print(f"  Change: {len(imp_solved) - len(div_solved):+d} ({imp_rate - div_rate:+.1f}%)")

    # Average accuracy
    div_avg = np.mean([r['accuracy'] for r in div_results]) * 100
    imp_avg = np.mean([r['accuracy'] for r in imp_results]) * 100

    print(f"\n📊 AVERAGE ACCURACY:")
    print(f"  Diverse:  {div_avg:.1f}%")
    print(f"  Improved: {imp_avg:.1f}%")
    print(f"  Change: {imp_avg - div_avg:+.1f}%")

    # Hypothesis counts
    div_hyps = np.mean([r['hyp_count'] for r in div_results])
    imp_hyps = np.mean([r['hyp_count'] for r in imp_results])

    print(f"\n🔍 HYPOTHESES PER TASK:")
    print(f"  Diverse:  {div_hyps:.1f}")
    print(f"  Improved: {imp_hyps:.1f}")

    # Improvements
    improved_tasks = sum(1 for i in range(len(task_files))
                        if imp_results[i]['accuracy'] > div_results[i]['accuracy'])

    print(f"\n📈 TASKS IMPROVED: {improved_tasks}/{len(task_files)}")

    if newly_solved:
        print(f"\n🎉 NEWLY SOLVED:")
        for task_id in newly_solved:
            idx = next(i for i, r in enumerate(div_results) if r['task_id'] == task_id)
            div_acc = div_results[idx]['accuracy'] * 100
            print(f"  - {task_id}: {div_acc:.1f}% → 100% ✓")

    # Top improvements
    improvements = []
    for i in range(len(task_files)):
        diff = imp_results[i]['accuracy'] - div_results[i]['accuracy']
        if diff > 0.05:
            improvements.append({
                'task': imp_results[i]['task_id'],
                'before': div_results[i]['accuracy'] * 100,
                'after': imp_results[i]['accuracy'] * 100,
                'improvement': diff * 100
            })

    if improvements:
        improvements.sort(key=lambda x: x['improvement'], reverse=True)
        print(f"\n📈 TOP IMPROVEMENTS:")
        for imp in improvements[:5]:
            print(f"  {imp['task']}: {imp['before']:.1f}% → {imp['after']:.1f}% (+{imp['improvement']:.1f}%)")

    print(f"\n{'='*70}")
    if imp_rate > div_rate:
        print("✅ IMPROVEMENT CONFIRMED")
    elif imp_avg > div_avg + 2:
        print("📊 ACCURACY IMPROVEMENT")
    else:
        print("⚠️  No significant improvement yet")
    print("="*70)


if __name__ == '__main__':
    main()
