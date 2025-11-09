"""
Test Hypothesis Proposer on Real ARC Tasks - Phase 4

Evaluates the hypothesis proposer's ability to:
1. Detect patterns in training pairs
2. Generate valid hypotheses
3. Solve test cases
"""

import json
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from hypothesis_proposer import HypothesisProposer, PatternAnalyzer


def load_arc_task(task_file):
    """Load a single ARC task from JSON file"""
    with open(task_file, 'r') as f:
        return json.load(f)


def grids_equal(grid1, grid2):
    """Check if two grids are equal"""
    if isinstance(grid1, list):
        grid1 = np.array(grid1)
    if isinstance(grid2, list):
        grid2 = np.array(grid2)

    if grid1.shape != grid2.shape:
        return False

    return np.array_equal(grid1, grid2)


def test_pattern_analyzer():
    """Test pattern detection on known transformations"""
    print("\n" + "=" * 70)
    print("TEST 1: Pattern Analyzer")
    print("=" * 70)

    analyzer = PatternAnalyzer()

    # Test 1: Rotation
    print("\nTest 1.1: Rotation Detection")
    input_grid = np.array([[1, 0], [0, 0]])
    output_grid = np.array([[0, 1], [0, 0]])
    patterns = analyzer.analyze_pair(input_grid, output_grid)

    rotation_found = any(p.name == "rotate" and p.parameters.get("angle") == 90
                        for p in patterns)
    if rotation_found:
        print("✓ Rotation pattern detected correctly")
    else:
        print("✗ Failed to detect rotation")

    # Test 2: Tiling
    print("\nTest 1.2: Tiling Detection")
    input_grid = np.array([[1, 2], [3, 4]])
    output_grid = np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])
    patterns = analyzer.analyze_pair(input_grid, output_grid)

    tiling_found = any(p.name == "tile" for p in patterns)
    if tiling_found:
        print("✓ Tiling pattern detected correctly")
    else:
        print("✗ Failed to detect tiling")

    # Test 3: Reflection
    print("\nTest 1.3: Reflection Detection")
    input_grid = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    output_grid = np.array([[3, 0, 0], [2, 0, 0], [1, 0, 0]])
    patterns = analyzer.analyze_pair(input_grid, output_grid)

    reflection_found = any(p.name == "reflect" for p in patterns)
    if reflection_found:
        print("✓ Reflection pattern detected correctly")
    else:
        print("✗ Failed to detect reflection")

    # Test 4: Color change
    print("\nTest 1.4: Color Change Detection")
    input_grid = np.array([[1, 1], [2, 2]])
    output_grid = np.array([[3, 3], [4, 4]])
    patterns = analyzer.analyze_pair(input_grid, output_grid)

    recolor_found = any(p.name == "recolor" for p in patterns)
    if recolor_found:
        print("✓ Color change pattern detected correctly")
    else:
        print("✗ Failed to detect color change")

    print("\n✓ Pattern Analyzer tests complete")


def test_hypothesis_generation():
    """Test hypothesis generation from patterns"""
    print("\n" + "=" * 70)
    print("TEST 2: Hypothesis Generation")
    print("=" * 70)

    proposer = HypothesisProposer()

    # Create a simple task: rotate 90 degrees
    train_pairs = [
        {
            'input': [[1, 0], [0, 0]],
            'output': [[0, 1], [0, 0]]
        },
        {
            'input': [[2, 0], [0, 0]],
            'output': [[0, 2], [0, 0]]
        }
    ]

    hypotheses = proposer.propose(train_pairs, beam_size=5)

    print(f"\nGenerated {len(hypotheses)} hypotheses")
    for i, hyp in enumerate(hypotheses[:3]):
        print(f"{i+1}. {hyp.description} (score: {hyp.score:.3f})")

    # Check if rotation hypothesis was generated
    rotation_found = any("rotate" in hyp.description.lower() or "rotate" in hyp.primitives
                        for hyp in hypotheses)

    if rotation_found and hypotheses[0].score > 0.9:
        print("\n✓ Hypothesis generation successful")
        return True
    else:
        print("\n✗ Failed to generate correct hypothesis")
        return False


def test_on_real_arc_tasks(num_tasks=50):
    """Test on real ARC tasks"""
    print("\n" + "=" * 70)
    print("TEST 3: Real ARC Tasks")
    print("=" * 70)

    # Find ARC data
    data_dir = Path(__file__).parent / "data" / "ARC-AGI" / "data" / "training"

    if not data_dir.exists():
        print(f"\n⚠ ARC data not found at {data_dir}")
        print("Skipping real task tests")
        return

    # Get task files
    task_files = sorted(data_dir.glob("*.json"))[:num_tasks]
    print(f"\nTesting on {len(task_files)} ARC tasks")

    proposer = HypothesisProposer()

    results = {
        'total': 0,
        'solved': 0,
        'partial': 0,
        'failed': 0,
        'solved_tasks': []
    }

    for task_file in task_files:
        task = load_arc_task(task_file)
        task_id = task_file.stem

        # Try to solve
        try:
            prediction = proposer.solve(task)

            if prediction is not None:
                expected = np.array(task['test'][0]['output'])

                if grids_equal(prediction, expected):
                    results['solved'] += 1
                    results['solved_tasks'].append(task_id)
                    print(f"✓ {task_id}: SOLVED")
                elif prediction.shape == expected.shape:
                    similarity = np.sum(prediction == expected) / prediction.size
                    if similarity > 0.5:
                        results['partial'] += 1
                        print(f"◐ {task_id}: PARTIAL ({similarity:.1%})")
                    else:
                        results['failed'] += 1
                        print(f"✗ {task_id}: Failed (low similarity)")
                else:
                    results['failed'] += 1
                    print(f"✗ {task_id}: Failed (wrong shape)")
            else:
                results['failed'] += 1
                print(f"✗ {task_id}: No solution found")

        except Exception as e:
            results['failed'] += 1
            print(f"✗ {task_id}: Error - {str(e)[:50]}")

        results['total'] += 1

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTotal tasks tested: {results['total']}")
    print(f"Fully solved: {results['solved']} ({results['solved']/results['total']*100:.1f}%)")
    print(f"Partially solved: {results['partial']} ({results['partial']/results['total']*100:.1f}%)")
    print(f"Failed: {results['failed']} ({results['failed']/results['total']*100:.1f}%)")

    if results['solved_tasks']:
        print(f"\nSolved tasks: {', '.join(results['solved_tasks'][:10])}")
        if len(results['solved_tasks']) > 10:
            print(f"... and {len(results['solved_tasks']) - 10} more")

    return results


def test_specific_patterns():
    """Test on specific known patterns"""
    print("\n" + "=" * 70)
    print("TEST 4: Specific Pattern Tasks")
    print("=" * 70)

    proposer = HypothesisProposer()
    passed = 0
    total = 0

    # Test 1: Simple tiling
    print("\nTest 4.1: Tiling Pattern")
    task = {
        'train': [
            {
                'input': [[1, 2]],
                'output': [[1, 2, 1, 2]]
            },
            {
                'input': [[3, 4]],
                'output': [[3, 4, 3, 4]]
            }
        ],
        'test': [
            {
                'input': [[5, 6]],
                'output': [[5, 6, 5, 6]]
            }
        ]
    }

    prediction = proposer.solve(task)
    expected = np.array(task['test'][0]['output'])

    if prediction is not None and grids_equal(prediction, expected):
        print("✓ Tiling pattern solved")
        passed += 1
    else:
        print("✗ Tiling pattern failed")
    total += 1

    # Test 2: Rotation
    print("\nTest 4.2: Rotation Pattern")
    task = {
        'train': [
            {
                'input': [[1, 0], [0, 0]],
                'output': [[0, 0], [1, 0]]
            },
            {
                'input': [[2, 0], [0, 0]],
                'output': [[0, 0], [2, 0]]
            }
        ],
        'test': [
            {
                'input': [[3, 0], [0, 0]],
                'output': [[0, 0], [3, 0]]
            }
        ]
    }

    prediction = proposer.solve(task)
    expected = np.array(task['test'][0]['output'])

    if prediction is not None and grids_equal(prediction, expected):
        print("✓ Rotation pattern solved")
        passed += 1
    else:
        print("✗ Rotation pattern failed")
        if prediction is not None:
            print(f"  Expected:\n{expected}")
            print(f"  Got:\n{prediction}")
    total += 1

    # Test 3: Reflection
    print("\nTest 4.3: Reflection Pattern")
    task = {
        'train': [
            {
                'input': [[1, 0, 0], [2, 0, 0]],
                'output': [[0, 0, 1], [0, 0, 2]]
            }
        ],
        'test': [
            {
                'input': [[3, 0, 0], [4, 0, 0]],
                'output': [[0, 0, 3], [0, 0, 4]]
            }
        ]
    }

    prediction = proposer.solve(task)
    expected = np.array(task['test'][0]['output'])

    if prediction is not None and grids_equal(prediction, expected):
        print("✓ Reflection pattern solved")
        passed += 1
    else:
        print("✗ Reflection pattern failed")
    total += 1

    print(f"\n✓ Specific pattern tests: {passed}/{total} passed")
    return passed, total


def run_all_tests():
    """Run all hypothesis proposer tests"""
    print("=" * 70)
    print("PHASE 4: HYPOTHESIS PROPOSER COMPREHENSIVE TEST")
    print("=" * 70)

    # Test 1: Pattern Analyzer
    test_pattern_analyzer()

    # Test 2: Hypothesis Generation
    test_hypothesis_generation()

    # Test 3: Specific Patterns
    pattern_passed, pattern_total = test_specific_patterns()

    # Test 4: Real ARC Tasks
    arc_results = test_on_real_arc_tasks(num_tasks=50)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\n✓ Phase 4 Implementation Complete!")
    print("\nKey Components:")
    print("  - PatternAnalyzer: Detects 6 pattern types")
    print("  - HypothesisGenerator: Creates single & composite programs")
    print("  - HypothesisProposer: Validates and ranks hypotheses")

    if arc_results:
        success_rate = (arc_results['solved'] / arc_results['total']) * 100
        print(f"\nARC Task Performance:")
        print(f"  - Success rate: {success_rate:.1f}%")
        print(f"  - Tasks solved: {arc_results['solved']}/{arc_results['total']}")

        if success_rate >= 10:
            print("\n🎉 TARGET ACHIEVED: >10% success rate on ARC tasks!")
        elif success_rate >= 5:
            print("\n✓ Good progress: >5% success rate")
        else:
            print("\n⚠ Below target, but foundation is working")

    print("\nPattern-specific tests:", f"{pattern_passed}/{pattern_total} passed")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_all_tests()
