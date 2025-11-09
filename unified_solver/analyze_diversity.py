"""
Analyze why top-2 predictions are often the same
"""

import numpy as np
from arc_active_inference_solver import ARCActiveInferenceSolver, ARCTask, Grid


def analyze_prediction_diversity():
    """Investigate why predictions are the same"""

    print("="*80)
    print("ANALYZING PREDICTION DIVERSITY ISSUE")
    print("="*80)

    # Test case that produces same predictions
    task = ARCTask([
        (Grid([[1, 2], [3, 4]]), Grid([[2, 1], [4, 3]])),
        (Grid([[5, 6], [7, 8]]), Grid([[6, 5], [8, 7]])),
    ], Grid([[9, 1], [2, 3]]))

    solver = ARCActiveInferenceSolver()

    # Capture the solving process
    from arc_active_inference_solver import (
        PerceptionModule, HypothesisGenerator,
        ActiveInferenceEngine, StabilityFilter
    )

    perception = PerceptionModule()
    generator = HypothesisGenerator(perception)
    active_inference = ActiveInferenceEngine()
    stability_filter = StabilityFilter()

    # Get features
    first_input, first_output = task.train_pairs[0]
    features = perception.perceive(first_input)

    # Generate hypotheses
    hypotheses = generator.generate_hypotheses(task, features)
    print(f"\nGenerated {len(hypotheses)} hypotheses")

    # Initialize beliefs
    belief = active_inference.initialize_beliefs(hypotheses)

    # Update beliefs
    for idx, (input_grid, output_grid) in enumerate(task.train_pairs):
        belief = active_inference.update_beliefs(belief, (input_grid, output_grid), hypotheses)

    # Assess stability for all hypotheses
    for h in hypotheses:
        stability_filter.assess_stability(h, task, belief)

    # Final ranking
    final_scores = {}
    for h in hypotheses:
        posterior = belief.probabilities.get(h, 0.0)
        stability = belief.stability_scores.get(h, None)
        if stability is None:
            stability = 0.0
        final_scores[h] = posterior * stability

    # Get top-10 by score
    ranked = sorted(hypotheses, key=lambda h: final_scores.get(h, 0.0), reverse=True)
    top_10 = ranked[:10]

    print(f"\nTop 10 hypotheses by final score:")
    print(f"{'Rank':<5} {'Name':<30} {'Posterior':<12} {'Stability':<12} {'Score':<12}")
    print("-"*80)

    for i, h in enumerate(top_10, 1):
        posterior = belief.probabilities.get(h, 0.0)
        stability = belief.stability_scores.get(h, None) if h in belief.stability_scores else None
        if stability is None:
            stability = 0.0
        score = final_scores.get(h, 0.0)
        print(f"{i:<5} {h.name:<30} {posterior:<12.6f} {stability:<12.6f} {score:<12.9f}")

    # Apply top-2 to test input
    print(f"\n{'='*80}")
    print("PREDICTIONS ON TEST INPUT")
    print(f"{'='*80}")

    print("\nTest input:")
    print(task.test_input.data)

    for i, h in enumerate(top_10[:5], 1):
        pred = h.apply(task.test_input)
        print(f"\nPrediction {i} ({h.name}):")
        print(pred.data)

    # Check for duplicates
    predictions_map = {}
    for h in top_10[:5]:
        pred = h.apply(task.test_input)
        key = pred.data.tobytes()
        if key not in predictions_map:
            predictions_map[key] = []
        predictions_map[key].append(h.name)

    print(f"\n{'='*80}")
    print("DUPLICATE ANALYSIS")
    print(f"{'='*80}")

    print(f"\nUnique predictions: {len(predictions_map)}")
    for i, (key, names) in enumerate(predictions_map.items(), 1):
        if len(names) > 1:
            print(f"  Group {i}: {len(names)} hypotheses produce same output")
            for name in names:
                print(f"    - {name}")

    # Root cause analysis
    print(f"\n{'='*80}")
    print("ROOT CAUSE ANALYSIS")
    print(f"{'='*80}")

    # Check if top hypotheses have very different scores
    top_2_scores = [final_scores.get(h, 0.0) for h in ranked[:2]]
    score_ratio = top_2_scores[1] / top_2_scores[0] if top_2_scores[0] > 0 else 0

    print(f"\nTop-2 score ratio: {score_ratio:.6f}")
    print(f"Top score: {top_2_scores[0]:.9f}")
    print(f"2nd score: {top_2_scores[1]:.9f}")

    if score_ratio > 0.9:
        print("\n✗ ISSUE: Top-2 hypotheses have very similar scores")
        print("  This suggests weak differentiation between hypotheses")
    elif score_ratio < 0.1:
        print("\n✓ Top hypothesis strongly dominates")
    else:
        print("\n~ Moderate score separation")

    # Check if top-2 are actually different transforms
    if len(ranked) >= 2:
        h1, h2 = ranked[0], ranked[1]
        pred1 = h1.apply(task.test_input)
        pred2 = h2.apply(task.test_input)

        if np.array_equal(pred1.data, pred2.data):
            print(f"\n✗ PROBLEM: Top-2 hypotheses produce identical output!")
            print(f"  Hypothesis 1: {h1.name}")
            print(f"  Hypothesis 2: {h2.name}")
            print(f"  Both produce: {pred1.data.tolist()}")

            # Check if they're actually the same transformation
            if h1.name == h2.name:
                print("  → These are the SAME hypothesis (duplicate in list)")
            else:
                print("  → These are DIFFERENT hypotheses that happen to produce same output")
        else:
            print(f"\n✓ Top-2 hypotheses produce different outputs")


if __name__ == "__main__":
    analyze_prediction_diversity()
