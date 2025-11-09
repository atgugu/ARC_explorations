from arc_active_inference_solver import ARCActiveInferenceSolver, ARCTask, Grid
import numpy as np

# Test a case that we know produces same predictions (edge_single_pixel)
task = ARCTask([
    (Grid([[1]]), Grid([[1]])),
    (Grid([[2]]), Grid([[2]])),
], Grid([[3]]))

solver = ARCActiveInferenceSolver()
predictions = solver.solve(task, verbose=False)

print("Test input:", task.test_input.data)
print("\nPrediction 1:", predictions[0].data)
print("Prediction 2:", predictions[1].data)
print("\nAre they the same?", np.array_equal(predictions[0].data, predictions[1].data))

# Get the actual hypotheses
from arc_active_inference_solver import PerceptionModule, HypothesisGenerator, ActiveInferenceEngine, StabilityFilter

perception = PerceptionModule()
generator = HypothesisGenerator(perception)
active_inference = ActiveInferenceEngine()
stability_filter = StabilityFilter()

features = perception.perceive(task.train_pairs[0][0])
hypotheses = generator.generate_hypotheses(task, features)

belief = active_inference.initialize_beliefs(hypotheses)
for input_grid, output_grid in task.train_pairs:
    belief = active_inference.update_beliefs(belief, (input_grid, output_grid), hypotheses)

for h in hypotheses:
    stability_filter.assess_stability(h, task, belief)

final_scores = {}
for h in hypotheses:
    posterior = belief.probabilities.get(h, 0.0)
    stability = belief.stability_scores.get(h, None)
    if stability is None:
        stability = 0.0
    final_scores[h] = posterior * stability

ranked = sorted(hypotheses, key=lambda h: final_scores.get(h, 0.0), reverse=True)

print(f"\nTop 5 hypotheses:")
for i, h in enumerate(ranked[:5], 1):
    score = final_scores.get(h, 0.0)
    pred = h.apply(task.test_input)
    print(f"{i}. {h.name:30s} score={score:.9f} -> {pred.data.tolist()}")
