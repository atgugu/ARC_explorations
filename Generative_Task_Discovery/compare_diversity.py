"""
Comprehensive Comparison: Original vs Diverse Solver

Tests both solvers on the same 15 tasks and compares:
- Diversity rate (% predictions differ)
- Success rate (% tasks solved)
- Pred2 saves (cases where pred2 correct but pred1 wrong)
"""

import numpy as np
from typing import Dict, List, Any
from arc_generative_solver import ARCGenerativeSolver
from diverse_solver import DiverseARCSolver, compare_solvers
from arc_test_suite import ARCTestSuite


class DiversityComparison:
    """Compare original vs diverse solver comprehensively"""

    def __init__(self):
        self.original_solver = ARCGenerativeSolver(
            max_candidates=100,
            beam_width=15,
            active_inference_steps=5
        )

        self.diverse_solvers = {
            "schema_first": DiverseARCSolver(
                max_candidates=100,
                beam_width=15,
                active_inference_steps=5,
                diversity_strategy="schema_first"
            ),
            "stochastic": DiverseARCSolver(
                max_candidates=100,
                beam_width=15,
                active_inference_steps=5,
                diversity_strategy="stochastic"
            ),
            "hybrid": DiverseARCSolver(
                max_candidates=100,
                beam_width=15,
                active_inference_steps=5,
                diversity_strategy="hybrid"
            )
        }

        self.test_suite = ARCTestSuite(self.original_solver)
        self.tasks = self.test_suite.create_diverse_test_tasks()

    def run_comparison(self) -> Dict[str, Any]:
        """Run full comparison across all strategies"""

        print("\n" + "="*70)
        print("COMPREHENSIVE DIVERSITY COMPARISON")
        print("="*70)
        print(f"\nTesting {len(self.tasks)} tasks across multiple strategies")

        results = {
            "original": self._test_solver(
                self.original_solver,
                "Original (Top-2 by Probability)"
            )
        }

        for strategy_name, solver in self.diverse_solvers.items():
            results[strategy_name] = self._test_solver(
                solver,
                f"Diverse ({strategy_name})"
            )

        # Generate comparison report
        self._print_comparison_report(results)

        return results

    def _test_solver(self, solver, solver_name: str) -> Dict[str, Any]:
        """Test a solver on all tasks"""

        print(f"\n{'='*70}")
        print(f"Testing: {solver_name}")
        print(f"{'='*70}")

        results = []
        n_diverse = 0
        n_success = 0
        n_pred2_saves = 0

        for i, task_data in enumerate(self.tasks, 1):
            task = task_data["task"]
            name = task_data["name"]

            print(f"\n[{i}/{len(self.tasks)}] {name}...", end=" ")

            try:
                pred1, pred2, metadata = solver.solve(task)
                target = np.array(task["test"][0]["output"])

                from arc_generative_solver import evaluate_predictions
                eval_res = evaluate_predictions(pred1, pred2, target)

                diverse = not np.array_equal(pred1, pred2)
                pred2_saves = eval_res['exact_match_2'] and not eval_res['exact_match_1']

                if diverse:
                    n_diverse += 1
                if eval_res['any_correct']:
                    n_success += 1
                if pred2_saves:
                    n_pred2_saves += 1

                status = "✓" if eval_res['any_correct'] else "✗"
                div_mark = "D" if diverse else "S"  # D=Diverse, S=Same
                save_mark = "!" if pred2_saves else ""

                print(f"{status} {div_mark}{save_mark}")

                results.append({
                    "name": name,
                    "diverse": diverse,
                    "success": eval_res['any_correct'],
                    "pred1_correct": eval_res['exact_match_1'],
                    "pred2_correct": eval_res['exact_match_2'],
                    "pred2_saves": pred2_saves,
                    "pred1_acc": eval_res['pixel_accuracy_1'],
                    "pred2_acc": eval_res['pixel_accuracy_2']
                })

            except Exception as e:
                print(f"✗ ERROR: {e}")
                results.append({
                    "name": name,
                    "error": str(e)
                })

        # Summary
        total = len(self.tasks)
        diversity_rate = n_diverse / total
        success_rate = n_success / total

        print(f"\n{'='*70}")
        print(f"Summary for {solver_name}")
        print(f"{'='*70}")
        print(f"Diversity Rate: {n_diverse}/{total} ({diversity_rate:.1%})")
        print(f"Success Rate: {n_success}/{total} ({success_rate:.1%})")
        print(f"Pred2 Saves: {n_pred2_saves} tasks")

        return {
            "solver_name": solver_name,
            "results": results,
            "n_diverse": n_diverse,
            "n_success": n_success,
            "n_pred2_saves": n_pred2_saves,
            "diversity_rate": diversity_rate,
            "success_rate": success_rate,
            "total": total
        }

    def _print_comparison_report(self, results: Dict[str, Any]):
        """Print comprehensive comparison report"""

        print("\n" + "="*70)
        print("COMPARISON REPORT")
        print("="*70)

        # Overall metrics table
        print("\n" + "="*70)
        print("OVERALL METRICS")
        print("="*70)

        print(f"\n{'Strategy':<30} {'Diversity':<15} {'Success':<15} {'Pred2 Saves'}")
        print("-"*70)

        for strategy, data in results.items():
            diversity = f"{data['n_diverse']}/{data['total']} ({data['diversity_rate']:.1%})"
            success = f"{data['n_success']}/{data['total']} ({data['success_rate']:.1%})"
            saves = f"{data['n_pred2_saves']}"

            print(f"{strategy:<30} {diversity:<15} {success:<15} {saves}")

        # Improvement analysis
        print("\n" + "="*70)
        print("IMPROVEMENT ANALYSIS")
        print("="*70)

        original = results["original"]

        for strategy, data in results.items():
            if strategy == "original":
                continue

            print(f"\n{strategy.upper()} vs Original:")

            # Diversity improvement
            div_improvement = data['diversity_rate'] - original['diversity_rate']
            print(f"  Diversity: {original['diversity_rate']:.1%} → {data['diversity_rate']:.1%} "
                  f"({div_improvement:+.1%})")

            # Success rate change
            success_change = data['success_rate'] - original['success_rate']
            print(f"  Success: {original['success_rate']:.1%} → {data['success_rate']:.1%} "
                  f"({success_change:+.1%})")

            # Pred2 saves improvement
            saves_improvement = data['n_pred2_saves'] - original['n_pred2_saves']
            print(f"  Pred2 Saves: {original['n_pred2_saves']} → {data['n_pred2_saves']} "
                  f"({saves_improvement:+d})")

        # Task-by-task where diversity helped
        print("\n" + "="*70)
        print("TASKS WHERE DIVERSE PREDICTIONS HELPED")
        print("="*70)

        for strategy, data in results.items():
            if strategy == "original":
                continue

            helped_tasks = [
                r for r in data['results']
                if r.get('pred2_saves', False)
            ]

            if helped_tasks:
                print(f"\n{strategy.upper()}:")
                for task in helped_tasks:
                    print(f"  ✓ {task['name']}")
                    print(f"    Pred1: {task['pred1_acc']:.1%}, Pred2: {task['pred2_acc']:.1%}")

        # Best strategy recommendation
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)

        # Find strategy with best balance of diversity and success
        best_strategy = None
        best_score = 0

        for strategy, data in results.items():
            if strategy == "original":
                continue

            # Score = diversity + success + 2*saves (saves are most valuable)
            score = (data['diversity_rate'] +
                    data['success_rate'] +
                    2 * data['n_pred2_saves'] / data['total'])

            if score > best_score:
                best_score = score
                best_strategy = strategy

        if best_strategy:
            print(f"\n✓ Best Strategy: {best_strategy.upper()}")
            data = results[best_strategy]
            print(f"  - Diversity: {data['diversity_rate']:.1%}")
            print(f"  - Success: {data['success_rate']:.1%}")
            print(f"  - Pred2 Saves: {data['n_pred2_saves']} tasks")

            # Specific improvements
            orig = results['original']
            div_gain = data['diversity_rate'] - orig['diversity_rate']
            saves_gain = data['n_pred2_saves'] - orig['n_pred2_saves']

            print(f"\n  Improvements over original:")
            print(f"  - Diversity: +{div_gain:.1%}")
            print(f"  - Pred2 Saves: +{saves_gain} tasks")

        # Summary
        print("\n" + "="*70)
        print("KEY TAKEAWAYS")
        print("="*70)

        orig_data = results['original']
        best_data = results[best_strategy] if best_strategy else orig_data

        print(f"\nOriginal Approach:")
        print(f"  • {orig_data['diversity_rate']:.1%} diversity rate")
        print(f"  • {orig_data['n_pred2_saves']} tasks saved by pred2")

        if best_strategy:
            print(f"\n{best_strategy.upper()} Approach:")
            print(f"  • {best_data['diversity_rate']:.1%} diversity rate")
            print(f"  • {best_data['n_pred2_saves']} tasks saved by pred2")

            if best_data['diversity_rate'] >= 0.95:
                print(f"\n✓ GOAL ACHIEVED: Near-100% diversity!")
            else:
                print(f"\n⚠ Diversity improved but not yet 100%")

            if best_data['n_pred2_saves'] > orig_data['n_pred2_saves']:
                print(f"✓ Pred2 saves more tasks than original!")


def main():
    """Run comprehensive comparison"""

    print("="*70)
    print("DIVERSITY STRATEGY EVALUATION")
    print("="*70)
    print("\nComparing strategies:")
    print("1. Original: Top-2 by probability")
    print("2. Schema-First: Force different schemas")
    print("3. Stochastic: Sample from posterior")
    print("4. Hybrid: Combine schema + stochastic")

    comparison = DiversityComparison()
    results = comparison.run_comparison()

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    return results


if __name__ == "__main__":
    results = main()
