"""Test Phase 7: Multi-Stage Pipelines vs Phase 6.1

Compare performance:
- Phase 6.1: Single-stage transformations (baseline)
- Phase 7: Multi-stage pipelines for sequential reasoning
"""

import json
import numpy as np
from pathlib import Path
import time

from arc_curiosity_solver.solver_conditional import ConditionalARCCuriositySolver


def test_solver(solver, task_data, task_id):
    """Test solver on task."""
    try:
        test_input = np.array(task_data['test'][0]['input'])
        expected = np.array(task_data['test'][0]['output'])

        train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                      for ex in task_data['train']]

        solver.verbose = False
        hypotheses = solver._generate_hypotheses(train_pairs, test_input)

        if not hypotheses:
            return {'solved': False, 'accuracy': 0.0, 'hyp_count': 0, 'task_id': task_id,
                   'pipeline_count': 0}

        best_acc = 0.0
        solved = False
        best_hyp_rank = -1
        best_hyp_type = 'unknown'
        pipeline_hyps = [h for h in hypotheses if 'pipeline' in h.name]

        for rank, h in enumerate(hypotheses[:10]):
            try:
                pred = h.program.function(test_input.copy())
                if np.array_equal(pred, expected):
                    solved = True
                    best_acc = 1.0
                    best_hyp_rank = rank
                    best_hyp_type = 'pipeline' if 'pipeline' in h.name else 'single-stage'
                    break
                if pred.shape == expected.shape:
                    acc = (pred == expected).mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_hyp_rank = rank
                        best_hyp_type = 'pipeline' if 'pipeline' in h.name else 'single-stage'
            except:
                pass

        return {
            'solved': solved,
            'accuracy': best_acc,
            'hyp_count': len(hypotheses),
            'pipeline_count': len(pipeline_hyps),
            'task_id': task_id,
            'best_rank': best_hyp_rank,
            'best_type': best_hyp_type
        }
    except Exception as e:
        return {
            'solved': False,
            'accuracy': 0.0,
            'hyp_count': 0,
            'pipeline_count': 0,
            'task_id': task_id,
            'error': str(e),
            'best_rank': -1,
            'best_type': 'error'
        }


def main():
    print("\n" + "="*80)
    print("PHASE 7 EVALUATION: MULTI-STAGE PIPELINES")
    print("="*80)
    print("\nComparing:")
    print("  Phase 6.1: Single-stage transformations (baseline: 57.6%)")
    print("  Phase 7:   Multi-stage pipelines for sequential reasoning")
    print("\n" + "="*80)

    training_dir = Path("ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))[:100]

    # Phase 6.1: No pipelines
    print(f"\n[1/2] Testing Phase 6.1 (single-stage only)...")
    phase61 = ConditionalARCCuriositySolver()
    phase61.use_composite_actions = True
    phase61.use_action_learning = True
    phase61.use_multi_stage_pipelines = False  # Disable pipelines
    phase61.validation_threshold = 0.15

    results_61 = []
    start_time = time.time()

    for i, task_file in enumerate(task_files):
        with open(task_file, 'r') as f:
            task_data = json.load(f)

        result = test_solver(phase61, task_data, task_file.stem)
        results_61.append(result)

        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/100")

    time_61 = time.time() - start_time

    # Phase 7: With pipelines
    print(f"\n[2/2] Testing Phase 7 (with multi-stage pipelines)...")
    phase7 = ConditionalARCCuriositySolver()
    phase7.use_composite_actions = True
    phase7.use_action_learning = True
    phase7.use_multi_stage_pipelines = True  # Enable pipelines
    phase7.validation_threshold = 0.15

    results_7 = []
    start_time = time.time()

    for i, task_file in enumerate(task_files):
        with open(task_file, 'r') as f:
            task_data = json.load(f)

        result = test_solver(phase7, task_data, task_file.stem)
        results_7.append(result)

        # Report pipeline wins
        r61 = results_61[i]
        if result['solved'] and not r61['solved'] and result['best_type'] == 'pipeline':
            print(f"  🎉 {result['task_id']}: NEW SOLVE by pipeline!")
        elif result['accuracy'] > r61['accuracy'] + 0.1 and result['best_type'] == 'pipeline':
            print(f"  📈 {result['task_id']}: Pipeline boost {r61['accuracy']*100:.1f}% → {result['accuracy']*100:.1f}%")

        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/100")

    time_7 = time.time() - start_time

    # ========= RESULTS =========
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Solve rates
    solved_61 = sum(1 for r in results_61 if r['solved'])
    solved_7 = sum(1 for r in results_7 if r['solved'])

    print(f"\n🎯 EXACT SOLVE RATE:")
    print(f"  Phase 6.1:  {solved_61}/100 ({solved_61}%)")
    print(f"  Phase 7:    {solved_7}/100 ({solved_7}%)")
    print(f"  Change:     {solved_7 - solved_61:+d} ({(solved_7 - solved_61)}%)")

    # Average accuracy
    avg_61 = np.mean([r['accuracy'] for r in results_61]) * 100
    avg_7 = np.mean([r['accuracy'] for r in results_7]) * 100

    print(f"\n📊 AVERAGE ACCURACY:")
    print(f"  Phase 6.1:  {avg_61:.2f}%")
    print(f"  Phase 7:    {avg_7:.2f}%")
    print(f"  Change:     {avg_7 - avg_61:+.2f}%")

    # Hypothesis counts
    avg_hyps_61 = np.mean([r['hyp_count'] for r in results_61])
    avg_hyps_7 = np.mean([r['hyp_count'] for r in results_7])
    avg_pipelines = np.mean([r['pipeline_count'] for r in results_7])

    print(f"\n🔍 HYPOTHESES PER TASK:")
    print(f"  Phase 6.1 total:    {avg_hyps_61:.1f}")
    print(f"  Phase 7 total:      {avg_hyps_7:.1f}")
    print(f"  Phase 7 pipelines:  {avg_pipelines:.1f}")
    print(f"  Change:             {avg_hyps_7 - avg_hyps_61:+.1f}")

    # Pipeline effectiveness
    pipeline_wins = sum(1 for r in results_7
                       if r['best_type'] == 'pipeline' and r['accuracy'] > 0.5)
    pipeline_solves = sum(1 for r in results_7
                         if r['solved'] and r['best_type'] == 'pipeline')

    print(f"\n🔧 PIPELINE EFFECTIVENESS:")
    print(f"  Best hypothesis was pipeline: {pipeline_wins}/100 ({pipeline_wins}%)")
    print(f"  Solved by pipeline:           {pipeline_solves}/100 ({pipeline_solves}%)")

    # Task improvements
    improved = sum(1 for i in range(100)
                  if results_7[i]['accuracy'] > results_61[i]['accuracy'])
    regressed = sum(1 for i in range(100)
                   if results_7[i]['accuracy'] < results_61[i]['accuracy'])

    print(f"\n📈 TASK-LEVEL CHANGES:")
    print(f"  Improved:   {improved}/100 ({improved}%)")
    print(f"  Regressed:  {regressed}/100 ({regressed}%)")
    print(f"  Unchanged:  {100 - improved - regressed}/100")

    # Pipeline-specific improvements
    pipeline_improvements = []
    for i in range(100):
        if (results_7[i]['accuracy'] > results_61[i]['accuracy'] + 0.05 and
            results_7[i]['best_type'] == 'pipeline'):
            pipeline_improvements.append({
                'task': results_7[i]['task_id'],
                'phase61': results_61[i]['accuracy'] * 100,
                'phase7': results_7[i]['accuracy'] * 100,
                'improvement': (results_7[i]['accuracy'] - results_61[i]['accuracy']) * 100
            })

    if pipeline_improvements:
        pipeline_improvements.sort(key=lambda x: x['improvement'], reverse=True)
        print(f"\n🚀 TOP PIPELINE IMPROVEMENTS:")
        for imp in pipeline_improvements[:10]:
            print(f"  {imp['task']}: {imp['phase61']:.1f}% → {imp['phase7']:.1f}% (+{imp['improvement']:.1f}%)")

    # Performance
    print(f"\n⏱️  PERFORMANCE:")
    print(f"  Phase 6.1 time: {time_61:.1f}s ({time_61/100:.2f}s per task)")
    print(f"  Phase 7 time:   {time_7:.1f}s ({time_7/100:.2f}s per task)")
    print(f"  Overhead:       {time_7 - time_61:.1f}s ({(time_7/time_61 - 1)*100:+.1f}%)")

    print("\n" + "="*80)

    # Verdict
    gain = avg_7 - avg_61

    if solved_7 > solved_61:
        print(f"✅ PHASE 7 SUCCESS: +{solved_7 - solved_61} new exact solve(s)!")
    elif gain > 5:
        print(f"✅ PHASE 7 SUCCESS: +{gain:.2f}% accuracy gain!")
    elif gain > 2:
        print(f"📊 PHASE 7 IMPROVEMENT: +{gain:.2f}% accuracy gain")
    elif gain > 0:
        print(f"🟡 PHASE 7 MODEST IMPROVEMENT: +{gain:.2f}% accuracy gain")
    elif gain > -1:
        print(f"🟡 PHASE 7 NEUTRAL: {gain:+.2f}% (within margin)")
    else:
        print(f"⚠️  PHASE 7 REGRESSION: {gain:.2f}% accuracy loss")

    if pipeline_wins > 10:
        print(f"✅ PIPELINES EFFECTIVE: Best hypothesis in {pipeline_wins}% of tasks")
    elif pipeline_wins > 5:
        print(f"🟡 PIPELINES MODERATELY USEFUL: Best in {pipeline_wins}% of tasks")
    else:
        print(f"⚠️  PIPELINES RARELY BEST: Only {pipeline_wins}% of tasks")

    print("="*80)

    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"  Phase 7 changes vs Phase 6.1:")
    print(f"    Accuracy:       {avg_7 - avg_61:+.2f}% ({avg_7:.2f}% vs {avg_61:.2f}%)")
    print(f"    Solve rate:     {solved_7 - solved_61:+d} ({solved_7} vs {solved_61})")
    print(f"    Pipeline wins:  {pipeline_wins}% of tasks")
    print(f"    Runtime:        {time_7:.1f}s vs {time_61:.1f}s")
    print()

    # Save results
    output_file = "phase7_100task_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'phase61': {
                'avg_accuracy': avg_61 / 100,
                'solved': solved_61,
                'avg_hypotheses': avg_hyps_61,
                'runtime': time_61
            },
            'phase7': {
                'avg_accuracy': avg_7 / 100,
                'solved': solved_7,
                'avg_hypotheses': avg_hyps_7,
                'avg_pipelines': avg_pipelines,
                'pipeline_wins': pipeline_wins,
                'runtime': time_7
            },
            'results': {
                'phase61': results_61,
                'phase7': results_7
            }
        }, f, indent=2)

    print(f"Detailed results saved to: {output_file}\n")


if __name__ == '__main__':
    main()
