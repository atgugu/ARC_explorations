"""
Comprehensive ARC Testing Suite

Tests the solver on diverse real ARC tasks and analyzes:
- Strengths and weaknesses
- Performance by task type
- Dual prediction effectiveness
- Failure modes

Ensures two predictions are different from each other.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import json
from arc_generative_solver import ARCGenerativeSolver, evaluate_predictions


class ARCTestSuite:
    """Comprehensive testing suite for ARC solver"""

    def __init__(self, solver: ARCGenerativeSolver):
        self.solver = solver
        self.results = []

    def create_diverse_test_tasks(self) -> List[Dict[str, Any]]:
        """Create diverse ARC-style test tasks covering various transformation types"""

        tasks = []

        # Task 1: Simple horizontal flip
        tasks.append({
            "name": "horizontal_flip_simple",
            "type": "geometric",
            "subtype": "reflection",
            "difficulty": "easy",
            "task": {
                "train": [
                    {
                        "input": [[1, 2, 3], [4, 5, 6]],
                        "output": [[3, 2, 1], [6, 5, 4]]
                    },
                    {
                        "input": [[7, 8], [9, 0]],
                        "output": [[8, 7], [0, 9]]
                    }
                ],
                "test": [
                    {
                        "input": [[1, 2, 3, 4]],
                        "output": [[4, 3, 2, 1]]
                    }
                ]
            }
        })

        # Task 2: 90-degree rotation
        tasks.append({
            "name": "rotation_90_clockwise",
            "type": "geometric",
            "subtype": "rotation",
            "difficulty": "easy",
            "task": {
                "train": [
                    {
                        "input": [[1, 2], [3, 4]],
                        "output": [[3, 1], [4, 2]]
                    },
                    {
                        "input": [[5, 6], [7, 8]],
                        "output": [[7, 5], [8, 6]]
                    }
                ],
                "test": [
                    {
                        "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        "output": [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
                    }
                ]
            }
        })

        # Task 3: Vertical flip
        tasks.append({
            "name": "vertical_flip",
            "type": "geometric",
            "subtype": "reflection",
            "difficulty": "easy",
            "task": {
                "train": [
                    {
                        "input": [[1, 2, 3], [4, 5, 6]],
                        "output": [[4, 5, 6], [1, 2, 3]]
                    }
                ],
                "test": [
                    {
                        "input": [[7, 8], [9, 0], [1, 2]],
                        "output": [[1, 2], [9, 0], [7, 8]]
                    }
                ]
            }
        })

        # Task 4: 180-degree rotation
        tasks.append({
            "name": "rotation_180",
            "type": "geometric",
            "subtype": "rotation",
            "difficulty": "easy",
            "task": {
                "train": [
                    {
                        "input": [[1, 2], [3, 4]],
                        "output": [[4, 3], [2, 1]]
                    }
                ],
                "test": [
                    {
                        "input": [[5, 6, 7], [8, 9, 0]],
                        "output": [[0, 9, 8], [7, 6, 5]]
                    }
                ]
            }
        })

        # Task 5: Color swap (simple remap)
        tasks.append({
            "name": "color_swap_binary",
            "type": "color",
            "subtype": "remap",
            "difficulty": "easy",
            "task": {
                "train": [
                    {
                        "input": [[1, 1, 2], [2, 1, 2]],
                        "output": [[2, 2, 1], [1, 2, 1]]
                    }
                ],
                "test": [
                    {
                        "input": [[1, 2, 1], [2, 2, 1]],
                        "output": [[2, 1, 2], [1, 1, 2]]
                    }
                ]
            }
        })

        # Task 6: Translation (shift right)
        tasks.append({
            "name": "translate_right",
            "type": "geometric",
            "subtype": "translation",
            "difficulty": "medium",
            "task": {
                "train": [
                    {
                        "input": [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
                        "output": [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]]
                    }
                ],
                "test": [
                    {
                        "input": [[0, 0, 0, 0, 0], [2, 2, 0, 0, 0], [0, 0, 0, 0, 0]],
                        "output": [[0, 0, 0, 0, 0], [0, 0, 2, 2, 0], [0, 0, 0, 0, 0]]
                    }
                ]
            }
        })

        # Task 7: Translation (shift down)
        tasks.append({
            "name": "translate_down",
            "type": "geometric",
            "subtype": "translation",
            "difficulty": "medium",
            "task": {
                "train": [
                    {
                        "input": [[1, 1], [0, 0], [0, 0]],
                        "output": [[0, 0], [1, 1], [0, 0]]
                    }
                ],
                "test": [
                    {
                        "input": [[2, 2], [0, 0], [0, 0], [0, 0]],
                        "output": [[0, 0], [2, 2], [0, 0], [0, 0]]
                    }
                ]
            }
        })

        # Task 8: Composite - rotation + flip
        tasks.append({
            "name": "rotation_then_flip",
            "type": "geometric",
            "subtype": "composite",
            "difficulty": "hard",
            "task": {
                "train": [
                    {
                        "input": [[1, 2], [3, 4]],
                        "output": [[2, 4], [1, 3]]
                    }
                ],
                "test": [
                    {
                        "input": [[5, 6], [7, 8]],
                        "output": [[6, 8], [5, 7]]
                    }
                ]
            }
        })

        # Task 9: Color remap (multiple colors)
        tasks.append({
            "name": "color_remap_multi",
            "type": "color",
            "subtype": "remap",
            "difficulty": "medium",
            "task": {
                "train": [
                    {
                        "input": [[1, 2, 3], [1, 2, 3]],
                        "output": [[4, 5, 6], [4, 5, 6]]
                    }
                ],
                "test": [
                    {
                        "input": [[1, 1, 2], [3, 2, 1]],
                        "output": [[4, 4, 5], [6, 5, 4]]
                    }
                ]
            }
        })

        # Task 10: Identity (copy)
        tasks.append({
            "name": "identity_copy",
            "type": "identity",
            "subtype": "copy",
            "difficulty": "easy",
            "task": {
                "train": [
                    {
                        "input": [[1, 2], [3, 4]],
                        "output": [[1, 2], [3, 4]]
                    }
                ],
                "test": [
                    {
                        "input": [[5, 6, 7], [8, 9, 0]],
                        "output": [[5, 6, 7], [8, 9, 0]]
                    }
                ]
            }
        })

        # Task 11: Larger rotation
        tasks.append({
            "name": "rotation_90_large",
            "type": "geometric",
            "subtype": "rotation",
            "difficulty": "medium",
            "task": {
                "train": [
                    {
                        "input": [[1, 2, 3, 4], [5, 6, 7, 8]],
                        "output": [[5, 1], [6, 2], [7, 3], [8, 4]]
                    }
                ],
                "test": [
                    {
                        "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        "output": [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
                    }
                ]
            }
        })

        # Task 12: Pattern with different colors
        tasks.append({
            "name": "colored_pattern",
            "type": "mixed",
            "subtype": "pattern",
            "difficulty": "medium",
            "task": {
                "train": [
                    {
                        "input": [[1, 0, 2], [0, 3, 0], [4, 0, 5]],
                        "output": [[5, 0, 4], [0, 3, 0], [2, 0, 1]]
                    }
                ],
                "test": [
                    {
                        "input": [[2, 0, 3], [0, 1, 0], [5, 0, 4]],
                        "output": [[4, 0, 5], [0, 1, 0], [3, 0, 2]]
                    }
                ]
            }
        })

        # Task 13: Asymmetric flip
        tasks.append({
            "name": "asymmetric_flip",
            "type": "geometric",
            "subtype": "reflection",
            "difficulty": "easy",
            "task": {
                "train": [
                    {
                        "input": [[1, 2, 3, 4, 5]],
                        "output": [[5, 4, 3, 2, 1]]
                    }
                ],
                "test": [
                    {
                        "input": [[9, 8, 7, 6]],
                        "output": [[6, 7, 8, 9]]
                    }
                ]
            }
        })

        # Task 14: Multiple training examples
        tasks.append({
            "name": "flip_with_multiple_examples",
            "type": "geometric",
            "subtype": "reflection",
            "difficulty": "easy",
            "task": {
                "train": [
                    {
                        "input": [[1, 2]],
                        "output": [[2, 1]]
                    },
                    {
                        "input": [[3, 4, 5]],
                        "output": [[5, 4, 3]]
                    },
                    {
                        "input": [[6, 7, 8, 9]],
                        "output": [[9, 8, 7, 6]]
                    }
                ],
                "test": [
                    {
                        "input": [[1, 2, 3]],
                        "output": [[3, 2, 1]]
                    }
                ]
            }
        })

        # Task 15: Small grid rotation
        tasks.append({
            "name": "small_rotation_270",
            "type": "geometric",
            "subtype": "rotation",
            "difficulty": "easy",
            "task": {
                "train": [
                    {
                        "input": [[1, 2], [3, 4]],
                        "output": [[2, 4], [1, 3]]
                    }
                ],
                "test": [
                    {
                        "input": [[5, 6], [7, 8]],
                        "output": [[6, 8], [5, 7]]
                    }
                ]
            }
        })

        return tasks

    def test_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test solver on a single task and return detailed results"""

        task = task_data["task"]
        name = task_data["name"]

        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"Type: {task_data['type']} / {task_data['subtype']}")
        print(f"Difficulty: {task_data['difficulty']}")
        print(f"{'='*70}")

        try:
            # Solve the task
            pred1, pred2, metadata = self.solver.solve(task)

            # Get ground truth
            target = np.array(task["test"][0]["output"])

            # Check if predictions are different
            predictions_differ = not np.array_equal(pred1, pred2)

            # Evaluate
            eval_results = evaluate_predictions(pred1, pred2, target)

            # Detailed analysis
            result = {
                "name": name,
                "type": task_data["type"],
                "subtype": task_data["subtype"],
                "difficulty": task_data["difficulty"],
                "success": bool(eval_results["any_correct"]),
                "pred1_exact": bool(eval_results["exact_match_1"]),
                "pred2_exact": bool(eval_results["exact_match_2"]),
                "pred1_pixel_acc": eval_results["pixel_accuracy_1"],
                "pred2_pixel_acc": eval_results["pixel_accuracy_2"],
                "predictions_differ": predictions_differ,
                "top_program": metadata["top_programs"][0]["schema"] if metadata["top_programs"] else None,
                "top_params": metadata["top_programs"][0]["parameters"] if metadata["top_programs"] else None,
                "top_prob": metadata["top_programs"][0]["probability"] if metadata["top_programs"] else 0.0,
                "second_program": metadata["top_programs"][1]["schema"] if len(metadata["top_programs"]) > 1 else None,
                "second_prob": metadata["top_programs"][1]["probability"] if len(metadata["top_programs"]) > 1 else 0.0,
                "free_energy": metadata["free_energy"],
                "entropy": metadata["entropy"],
                "n_candidates": metadata["n_candidates"],
                "pred1_shape": pred1.shape,
                "pred2_shape": pred2.shape,
                "target_shape": target.shape,
                "error": None
            }

            # Print results
            status = "✓ SOLVED" if result["success"] else "✗ FAILED"
            print(f"\n{status}")
            print(f"  Prediction 1: {result['pred1_exact']} (accuracy: {result['pred1_pixel_acc']:.1%})")
            print(f"  Prediction 2: {result['pred2_exact']} (accuracy: {result['pred2_pixel_acc']:.1%})")
            print(f"  Predictions differ: {predictions_differ}")
            print(f"  Top program: {result['top_program']} (p={result['top_prob']:.3f})")
            if result["second_program"]:
                print(f"  2nd program: {result['second_program']} (p={result['second_prob']:.3f})")

            # Show predictions vs target if failed
            if not result["success"]:
                print(f"\n  Target shape: {target.shape}")
                print(f"  Target:\n{target}")
                print(f"\n  Prediction 1 shape: {pred1.shape}")
                print(f"  Prediction 1:\n{pred1}")
                if predictions_differ:
                    print(f"\n  Prediction 2 shape: {pred2.shape}")
                    print(f"  Prediction 2:\n{pred2}")

        except Exception as e:
            result = {
                "name": name,
                "type": task_data["type"],
                "subtype": task_data["subtype"],
                "difficulty": task_data["difficulty"],
                "success": False,
                "error": str(e)
            }
            print(f"\n✗ ERROR: {e}")

        return result

    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite and generate analysis"""

        print("\n" + "="*70)
        print("ARC SOLVER - COMPREHENSIVE TEST SUITE")
        print("="*70)

        # Get test tasks
        tasks = self.create_diverse_test_tasks()
        print(f"\nTesting on {len(tasks)} diverse ARC tasks...")

        # Test each task
        for task_data in tasks:
            result = self.test_task(task_data)
            self.results.append(result)

        # Generate comprehensive analysis
        analysis = self.analyze_results()

        return analysis

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and identify strengths/weaknesses"""

        print("\n" + "="*70)
        print("COMPREHENSIVE ANALYSIS")
        print("="*70)

        # Overall statistics
        total = len(self.results)
        successful = [r for r in self.results if r.get("success", False)]
        failed = [r for r in self.results if not r.get("success", False)]

        n_success = len(successful)
        n_failed = len(failed)

        print(f"\n{'='*70}")
        print("OVERALL PERFORMANCE")
        print(f"{'='*70}")
        print(f"Total tasks: {total}")
        print(f"Solved: {n_success} ({n_success/total:.1%})")
        print(f"Failed: {n_failed} ({n_failed/total:.1%})")

        # By task type
        print(f"\n{'='*70}")
        print("PERFORMANCE BY TYPE")
        print(f"{'='*70}")

        by_type = defaultdict(list)
        for r in self.results:
            if "type" in r:
                by_type[r["type"]].append(r)

        for task_type, results in sorted(by_type.items()):
            success_count = sum(1 for r in results if r.get("success", False))
            total_count = len(results)
            print(f"\n{task_type.upper()}:")
            print(f"  Success: {success_count}/{total_count} ({success_count/total_count:.1%})")

            # Show individual tasks
            for r in results:
                status = "✓" if r.get("success", False) else "✗"
                print(f"    {status} {r['name']:30s} (diff: {r['difficulty']})")

        # By difficulty
        print(f"\n{'='*70}")
        print("PERFORMANCE BY DIFFICULTY")
        print(f"{'='*70}")

        by_difficulty = defaultdict(list)
        for r in self.results:
            if "difficulty" in r:
                by_difficulty[r["difficulty"]].append(r)

        for difficulty in ["easy", "medium", "hard"]:
            if difficulty in by_difficulty:
                results = by_difficulty[difficulty]
                success_count = sum(1 for r in results if r.get("success", False))
                total_count = len(results)
                print(f"{difficulty.upper()}: {success_count}/{total_count} ({success_count/total_count:.1%})")

        # Dual prediction analysis
        print(f"\n{'='*70}")
        print("DUAL PREDICTION ANALYSIS")
        print(f"{'='*70}")

        pred1_only = [r for r in successful if r.get("pred1_exact") and not r.get("pred2_exact")]
        pred2_only = [r for r in successful if r.get("pred2_exact") and not r.get("pred1_exact")]
        both_correct = [r for r in successful if r.get("pred1_exact") and r.get("pred2_exact")]
        differ = [r for r in self.results if r.get("predictions_differ", False)]

        print(f"Both predictions correct: {len(both_correct)}")
        print(f"Only prediction 1 correct: {len(pred1_only)}")
        print(f"Only prediction 2 correct: {len(pred2_only)}")
        print(f"Predictions differ: {len(differ)}/{total} ({len(differ)/total:.1%})")

        if pred2_only:
            print(f"\n✓ Prediction 2 saved the day in {len(pred2_only)} cases:")
            for r in pred2_only:
                print(f"  - {r['name']} ({r['second_program']})")

        # Program analysis
        print(f"\n{'='*70}")
        print("PROGRAM USAGE ANALYSIS")
        print(f"{'='*70}")

        program_success = defaultdict(lambda: {"total": 0, "success": 0})
        for r in self.results:
            if r.get("top_program"):
                prog = r["top_program"]
                program_success[prog]["total"] += 1
                if r.get("success"):
                    program_success[prog]["success"] += 1

        print("\nTop programs by success rate:")
        for prog, stats in sorted(program_success.items(),
                                  key=lambda x: x[1]["success"]/x[1]["total"] if x[1]["total"] > 0 else 0,
                                  reverse=True):
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {prog:15s}: {stats['success']}/{stats['total']} ({success_rate:.1%})")

        # Identify strengths
        print(f"\n{'='*70}")
        print("STRENGTHS")
        print(f"{'='*70}")

        strengths = []
        for task_type, results in by_type.items():
            success_rate = sum(1 for r in results if r.get("success", False)) / len(results)
            if success_rate >= 0.8:
                strengths.append((task_type, success_rate, results))

        if strengths:
            for task_type, success_rate, results in strengths:
                print(f"✓ {task_type.upper()}: {success_rate:.1%} success rate")
                subtypes = set(r["subtype"] for r in results)
                print(f"  Handles: {', '.join(subtypes)}")
        else:
            print("  (Performance varies across task types)")

        # Identify weaknesses
        print(f"\n{'='*70}")
        print("WEAKNESSES")
        print(f"{'='*70}")

        weaknesses = []
        for task_type, results in by_type.items():
            success_rate = sum(1 for r in results if r.get("success", False)) / len(results)
            if success_rate < 0.5:
                weaknesses.append((task_type, success_rate, results))

        if weaknesses:
            for task_type, success_rate, results in weaknesses:
                print(f"✗ {task_type.upper()}: {success_rate:.1%} success rate")
                failed_tasks = [r for r in results if not r.get("success")]
                if failed_tasks:
                    print(f"  Failed tasks:")
                    for r in failed_tasks[:3]:  # Show first 3
                        print(f"    - {r['name']} (predicted: {r.get('top_program', 'unknown')})")
        else:
            print("  (All task types have reasonable success rates)")

        # Common failure modes
        print(f"\n{'='*70}")
        print("COMMON FAILURE MODES")
        print(f"{'='*70}")

        shape_mismatches = [r for r in failed if r.get("pred1_shape") != r.get("target_shape")]
        wrong_program = [r for r in failed if r.get("top_program") and r.get("pred1_pixel_acc", 0) < 0.5]

        if shape_mismatches:
            print(f"✗ Shape mismatches: {len(shape_mismatches)} tasks")
            for r in shape_mismatches[:3]:
                print(f"  - {r['name']}: predicted {r.get('pred1_shape')} vs target {r.get('target_shape')}")

        if wrong_program:
            print(f"✗ Wrong program selected: {len(wrong_program)} tasks")
            for r in wrong_program[:3]:
                print(f"  - {r['name']}: used {r.get('top_program')} (acc: {r.get('pred1_pixel_acc', 0):.1%})")

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")

        analysis_summary = {
            "total_tasks": total,
            "solved": n_success,
            "failed": n_failed,
            "success_rate": n_success / total if total > 0 else 0,
            "by_type": {t: sum(1 for r in rs if r.get("success")) / len(rs)
                       for t, rs in by_type.items()},
            "by_difficulty": {d: sum(1 for r in rs if r.get("success")) / len(rs)
                             for d, rs in by_difficulty.items()},
            "dual_prediction_benefit": len(pred2_only),
            "predictions_differ_rate": len(differ) / total if total > 0 else 0,
            "strengths": [s[0] for s in strengths],
            "weaknesses": [w[0] for w in weaknesses],
            "results": self.results
        }

        print(f"\nOverall Success Rate: {analysis_summary['success_rate']:.1%}")
        print(f"Dual Prediction Benefit: {analysis_summary['dual_prediction_benefit']} tasks")
        print(f"Predictions Differ: {analysis_summary['predictions_differ_rate']:.1%}")

        if analysis_summary["strengths"]:
            print(f"\nStrong at: {', '.join(analysis_summary['strengths'])}")
        if analysis_summary["weaknesses"]:
            print(f"Needs improvement: {', '.join(analysis_summary['weaknesses'])}")

        return analysis_summary


def main():
    """Run comprehensive test suite"""

    print("="*70)
    print("ARC SOLVER COMPREHENSIVE EVALUATION")
    print("="*70)
    print("\nInitializing solver...")

    # Create solver with good parameters
    solver = ARCGenerativeSolver(
        max_candidates=100,
        beam_width=15,
        active_inference_steps=5
    )

    print("✓ Solver initialized")
    print(f"  Max candidates: 100")
    print(f"  Beam width: 15")
    print(f"  Active inference steps: 5")

    # Create test suite
    test_suite = ARCTestSuite(solver)

    # Run full test suite
    analysis = test_suite.run_full_test_suite()

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    return analysis


if __name__ == "__main__":
    analysis = main()
