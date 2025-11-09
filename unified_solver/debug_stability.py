"""
Debug why stability scores are all 0.0
"""

import numpy as np
from arc_active_inference_solver import (
    ARCActiveInferenceSolver, ARCTask, Grid,
    PerceptionModule, HypothesisGenerator, ActiveInferenceEngine, StabilityFilter
)


def debug_simple_task():
    """Debug a simple flip_vertical task"""

    print("="*80)
    print("DEBUGGING STABILITY COMPUTATION")
    print("="*80)

    # Simple flip vertical task
    task = ARCTask(
        train_pairs=[
            (Grid([[1, 2], [3, 4]]), Grid([[2, 1], [4, 3]])),
            (Grid([[5, 6], [7, 8]]), Grid([[6, 5], [8, 7]])),
        ],
        test_input=Grid([[1, 0], [2, 3]])
    )

    # Initialize components
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

    # Update beliefs for each training example
    for idx, (input_grid, output_grid) in enumerate(task.train_pairs):
        print(f"\n{'='*80}")
        print(f"Training Example {idx + 1}")
        print(f"{'='*80}")

        # Show input and output
        print("\nInput:")
        print(input_grid.data)
        print("\nExpected Output:")
        print(output_grid.data)

        # Update beliefs
        belief = active_inference.update_beliefs(belief, (input_grid, output_grid), hypotheses)

        # Check top hypotheses
        print(f"\nTop 5 hypotheses by probability:")
        top_5 = belief.top_k(5)
        for h, p in top_5:
            print(f"  {h.name:30s}: p={p:.6f}")

            # Test this hypothesis
            pred = h.apply(input_grid)
            print(f"    Prediction: {pred.data.tolist()}")
            print(f"    Expected:   {output_grid.data.tolist()}")
            match = np.array_equal(pred.data, output_grid.data)
            print(f"    Match: {match}")

        # Now compute stability for top hypotheses
        print(f"\nStability scores:")
        for h, p in top_5:
            stability = stability_filter.assess_stability(h, task, belief)
            print(f"  {h.name:30s}: stability={stability:.6f}")

            # Manual stability check
            accuracies = []
            for inp, out in task.train_pairs:
                pred = h.apply(inp)
                if pred.shape == out.shape:
                    acc = np.sum(pred.data == out.data) / pred.data.size
                else:
                    acc = 0.0
                accuracies.append(acc)

            print(f"    Accuracies per example: {accuracies}")
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            manual_stability = mean_acc * np.exp(-std_acc)
            print(f"    Mean: {mean_acc:.3f}, Std: {std_acc:.3f}, Stability: {manual_stability:.6f}")

    # Final ranking
    print(f"\n{'='*80}")
    print(f"FINAL RANKING")
    print(f"{'='*80}")

    final_scores = {}
    for h in hypotheses[:10]:  # Top 10 only
        posterior = belief.probabilities.get(h, 0.0)
        stability = belief.stability_scores.get(h, None)
        if stability is None:
            stability = 0.0
        score = posterior * stability
        final_scores[h] = score

        if posterior > 0.01 or stability > 0.01:
            print(f"{h.name:30s}: P={posterior:.6f}, S={stability:.6f}, Score={score:.9f}")

    # Get top-2 by final score
    ranked = sorted(hypotheses, key=lambda h: final_scores.get(h, 0.0), reverse=True)
    top_2 = ranked[:2]

    print(f"\nTop-2 by final score:")
    for i, h in enumerate(top_2, 1):
        print(f"{i}. {h.name}")
        print(f"   P={belief.probabilities.get(h, 0.0):.6f}")
        print(f"   S={belief.stability_scores.get(h, 0.0):.6f}")
        print(f"   Score={final_scores.get(h, 0.0):.9f}")

    # Test predictions
    print(f"\n{'='*80}")
    print(f"PREDICTIONS")
    print(f"{'='*80}")

    print("\nTest input:")
    print(task.test_input.data)

    for i, h in enumerate(top_2, 1):
        pred = h.apply(task.test_input)
        print(f"\nPrediction {i} ({h.name}):")
        print(pred.data)


if __name__ == "__main__":
    debug_simple_task()
