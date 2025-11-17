"""Test Phase 6.1 on 100 Real ARC-AGI Tasks

Comprehensive evaluation on 100 training tasks to get robust statistics.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
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
            return {'solved': False, 'accuracy': 0.0, 'hyp_count': 0, 'task_id': task_id}

        best_acc = 0.0
        solved = False
        best_hyp_rank = -1

        for rank, h in enumerate(hypotheses[:10]):  # Test top 10
            try:
                pred = h.program.function(test_input.copy())
                if np.array_equal(pred, expected):
                    solved = True
                    best_acc = 1.0
                    best_hyp_rank = rank
                    break
                if pred.shape == expected.shape:
                    acc = (pred == expected).mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_hyp_rank = rank
            except:
                pass

        return {
            'solved': solved,
            'accuracy': best_acc,
            'hyp_count': len(hypotheses),
            'task_id': task_id,
            'best_rank': best_hyp_rank
        }
    except Exception as e:
        return {
            'solved': False,
            'accuracy': 0.0,
            'hyp_count': 0,
            'task_id': task_id,
            'error': str(e),
            'best_rank': -1
        }


def main():
    print("\n" + "="*80)
    print("PHASE 6.1 COMPREHENSIVE EVALUATION: 100 REAL ARC-AGI TASKS")
    print("="*80)
    print("\nTesting Phase 6.1 (Object-Aware Action Learning + Confidence Prioritization)")
    print("on 100 real ARC-AGI training tasks for robust performance metrics.")
    print("\n" + "="*80)

    training_dir = Path("ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))[:100]

    print(f"\nFound {len(task_files)} tasks to evaluate")

    # Create solver
    solver = ConditionalARCCuriositySolver()
    solver.use_composite_actions = True
    solver.use_action_learning = True   # Phase 6.1
    solver.validation_threshold = 0.15

    results = []

    print(f"\nRunning evaluation...\n")
    start_time = time.time()

    for i, task_file in enumerate(task_files):
        task_id = task_file.stem

        with open(task_file, 'r') as f:
            task_data = json.load(f)

        result = test_solver(solver, task_data, task_id)
        results.append(result)

        # Progress indicator
        if (i+1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i+1) / elapsed
            remaining = (len(task_files) - (i+1)) / rate
            print(f"  Progress: {i+1}/{len(task_files)} ({(i+1)/len(task_files)*100:.1f}%) - "
                  f"Elapsed: {elapsed:.0f}s, Remaining: ~{remaining:.0f}s")

        # Report notable solves
        if result['solved']:
            print(f"  ✓ SOLVED: {task_id} (accuracy: 100%, rank: {result['best_rank']})")

    elapsed_time = time.time() - start_time

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Solve statistics
    solved_tasks = [r for r in results if r['solved']]
    high_acc_tasks = [r for r in results if r['accuracy'] >= 0.8 and not r['solved']]
    medium_acc_tasks = [r for r in results if 0.5 <= r['accuracy'] < 0.8]
    low_acc_tasks = [r for r in results if 0.1 <= r['accuracy'] < 0.5]
    zero_acc_tasks = [r for r in results if r['accuracy'] < 0.1]

    print(f"\n🎯 EXACT SOLVE RATE:")
    print(f"  Solved: {len(solved_tasks)}/{len(results)} ({len(solved_tasks)/len(results)*100:.1f}%)")

    print(f"\n📊 ACCURACY DISTRIBUTION:")
    print(f"  Perfect (100%):      {len(solved_tasks):3d} ({len(solved_tasks)/len(results)*100:.1f}%)")
    print(f"  High (80-99%):       {len(high_acc_tasks):3d} ({len(high_acc_tasks)/len(results)*100:.1f}%)")
    print(f"  Medium (50-79%):     {len(medium_acc_tasks):3d} ({len(medium_acc_tasks)/len(results)*100:.1f}%)")
    print(f"  Low (10-49%):        {len(low_acc_tasks):3d} ({len(low_acc_tasks)/len(results)*100:.1f}%)")
    print(f"  Very Low (<10%):     {len(zero_acc_tasks):3d} ({len(zero_acc_tasks)/len(results)*100:.1f}%)")

    # Average accuracy
    avg_acc = np.mean([r['accuracy'] for r in results]) * 100
    median_acc = np.median([r['accuracy'] for r in results]) * 100
    std_acc = np.std([r['accuracy'] for r in results]) * 100

    print(f"\n📈 ACCURACY STATISTICS:")
    print(f"  Mean:     {avg_acc:.2f}%")
    print(f"  Median:   {median_acc:.2f}%")
    print(f"  Std Dev:  {std_acc:.2f}%")
    print(f"  Min:      {min(r['accuracy'] for r in results)*100:.2f}%")
    print(f"  Max:      {max(r['accuracy'] for r in results)*100:.2f}%")

    # Hypothesis statistics
    avg_hyps = np.mean([r['hyp_count'] for r in results])
    median_hyps = np.median([r['hyp_count'] for r in results])
    total_hyps = sum(r['hyp_count'] for r in results)

    print(f"\n🔍 HYPOTHESIS STATISTICS:")
    print(f"  Total generated:     {total_hyps:,}")
    print(f"  Average per task:    {avg_hyps:.1f}")
    print(f"  Median per task:     {median_hyps:.1f}")
    print(f"  Min per task:        {min(r['hyp_count'] for r in results)}")
    print(f"  Max per task:        {max(r['hyp_count'] for r in results)}")

    # Rank analysis (where was the best hypothesis found?)
    if solved_tasks:
        ranks = [r['best_rank'] for r in solved_tasks]
        print(f"\n🎯 SOLUTION RANKING (for solved tasks):")
        print(f"  Found in top 1:  {sum(1 for r in ranks if r == 0)} ({sum(1 for r in ranks if r == 0)/len(ranks)*100:.1f}%)")
        print(f"  Found in top 3:  {sum(1 for r in ranks if r < 3)} ({sum(1 for r in ranks if r < 3)/len(ranks)*100:.1f}%)")
        print(f"  Found in top 5:  {sum(1 for r in ranks if r < 5)} ({sum(1 for r in ranks if r < 5)/len(ranks)*100:.1f}%)")
        print(f"  Found in top 10: {sum(1 for r in ranks if r < 10)} ({sum(1 for r in ranks if r < 10)/len(ranks)*100:.1f}%)")
        print(f"  Average rank:    {np.mean(ranks):.2f}")

    # Performance metrics
    print(f"\n⏱️  PERFORMANCE:")
    print(f"  Total time:          {elapsed_time:.1f}s")
    print(f"  Time per task:       {elapsed_time/len(results):.2f}s")
    print(f"  Tasks per second:    {len(results)/elapsed_time:.2f}")

    # Top performers
    if solved_tasks:
        print(f"\n🏆 SOLVED TASKS ({len(solved_tasks)}):")
        for r in sorted(solved_tasks, key=lambda x: x['best_rank']):
            print(f"  {r['task_id']}: 100% (rank {r['best_rank']}, {r['hyp_count']} hypotheses)")

    # Near-misses (high accuracy but not perfect)
    if high_acc_tasks:
        print(f"\n📊 NEAR-MISSES (80-99% accuracy, {len(high_acc_tasks)} tasks):")
        for r in sorted(high_acc_tasks, key=lambda x: x['accuracy'], reverse=True)[:10]:
            print(f"  {r['task_id']}: {r['accuracy']*100:.1f}% (rank {r['best_rank']}, {r['hyp_count']} hypotheses)")

    # Worst performers
    print(f"\n📉 LOWEST ACCURACY (bottom 10):")
    worst = sorted(results, key=lambda x: x['accuracy'])[:10]
    for r in worst:
        print(f"  {r['task_id']}: {r['accuracy']*100:.1f}% ({r['hyp_count']} hypotheses)")

    # Errors
    errors = [r for r in results if 'error' in r]
    if errors:
        print(f"\n⚠️  ERRORS ({len(errors)} tasks):")
        for r in errors[:5]:
            print(f"  {r['task_id']}: {r['error']}")

    print("\n" + "="*80)

    # Final verdict
    print(f"\n📋 FINAL VERDICT:")
    print(f"  Average Accuracy:  {avg_acc:.2f}%")
    print(f"  Solve Rate:        {len(solved_tasks)/len(results)*100:.1f}% ({len(solved_tasks)}/{len(results)})")
    print(f"  High Quality:      {(len(solved_tasks) + len(high_acc_tasks))/len(results)*100:.1f}% (≥80% accuracy)")
    print(f"  Hypotheses/Task:   {avg_hyps:.1f}")
    print(f"  Total Runtime:     {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")

    if avg_acc >= 50:
        print(f"\n✅ STRONG PERFORMANCE: {avg_acc:.1f}% average accuracy on 100 tasks")
    elif avg_acc >= 40:
        print(f"\n📊 GOOD PERFORMANCE: {avg_acc:.1f}% average accuracy on 100 tasks")
    elif avg_acc >= 30:
        print(f"\n🟡 MODERATE PERFORMANCE: {avg_acc:.1f}% average accuracy on 100 tasks")
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT: {avg_acc:.1f}% average accuracy on 100 tasks")

    print("="*80 + "\n")

    # Save detailed results to JSON
    output_file = "phase6_1_100task_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_tasks': len(results),
                'solved': len(solved_tasks),
                'solve_rate': len(solved_tasks) / len(results),
                'avg_accuracy': avg_acc / 100,
                'median_accuracy': median_acc / 100,
                'avg_hypotheses': avg_hyps,
                'total_time': elapsed_time
            },
            'results': results
        }, f, indent=2)

    print(f"Detailed results saved to: {output_file}\n")


if __name__ == '__main__':
    main()
