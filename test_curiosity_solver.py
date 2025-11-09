"""
Test the ARC Curiosity-Driven Active Inference Solver

Tests on various types of ARC-like tasks to demonstrate:
1. Curiosity-driven exploration
2. Belief dynamics
3. Active inference during inference
4. Always produces 2 predictions
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/ARC_explorations')

from arc_curiosity_solver.solver import ARCCuriositySolver, visualize_grids


def create_test_tasks():
    """Create various ARC-like test tasks."""

    tasks = []

    # Task 1: Simple translation (translate right by 1)
    task1 = {
        'name': 'Translate Right',
        'train': [
            (np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
             np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])),
            (np.array([[2, 2, 0], [0, 0, 0], [0, 0, 0]]),
             np.array([[0, 2, 2], [0, 0, 0], [0, 0, 0]])),
        ],
        'test': np.array([[3, 3, 3], [0, 0, 0], [0, 0, 0]]),
        'expected': np.array([[0, 3, 3], [0, 0, 0], [0, 0, 0]])  # Note: should be wrapped
    }
    tasks.append(task1)

    # Task 2: Rotation (90 degrees)
    task2 = {
        'name': 'Rotate 90',
        'train': [
            (np.array([[1, 0], [0, 0]]),
             np.array([[0, 1], [0, 0]])),
            (np.array([[2, 0], [2, 0]]),
             np.array([[2, 2], [0, 0]])),
        ],
        'test': np.array([[3, 0], [3, 0]]),
        'expected': np.array([[3, 3], [0, 0]])
    }
    tasks.append(task2)

    # Task 3: Reflection (horizontal)
    task3 = {
        'name': 'Reflect Horizontal',
        'train': [
            (np.array([[1, 0, 0]]),
             np.array([[0, 0, 1]])),
            (np.array([[2, 2, 0]]),
             np.array([[0, 2, 2]])),
        ],
        'test': np.array([[3, 3, 3]]),
        'expected': np.array([[3, 3, 3]])
    }
    tasks.append(task3)

    # Task 4: Scaling (2x)
    task4 = {
        'name': 'Scale 2x',
        'train': [
            (np.array([[1, 0], [0, 0]]),
             np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])),
            (np.array([[2, 2], [0, 0]]),
             np.array([[2, 2, 2, 2], [2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0]])),
        ],
        'test': np.array([[3, 0], [0, 0]]),
        'expected': np.array([[3, 3, 0, 0], [3, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    }
    tasks.append(task4)

    return tasks


def test_solver_on_task(solver, task, verbose=True):
    """Test solver on a single task."""

    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing Task: {task['name']}")
        print(f"{'='*70}")

        print("\nTraining Examples:")
        for i, (inp, out) in enumerate(task['train']):
            print(f"\nExample {i+1}:")
            visualize_grids([inp, out], ["Input", "Output"])

        print("\nTest Input:")
        visualize_grids([task['test']], ["Test"])

    # Solve
    pred1, pred2 = solver.solve(task['train'], task['test'], verbose=verbose)

    if verbose:
        print("\n" + "="*70)
        print("PREDICTIONS:")
        print("="*70)
        visualize_grids([pred1, pred2], ["Prediction 1", "Prediction 2"])

        if 'expected' in task:
            print("\nExpected:")
            visualize_grids([task['expected']], ["Expected Output"])

            # Check accuracy
            if pred1.shape == task['expected'].shape:
                accuracy1 = np.sum(pred1 == task['expected']) / task['expected'].size
                print(f"\nPrediction 1 Accuracy: {accuracy1*100:.1f}%")

            if pred2.shape == task['expected'].shape:
                accuracy2 = np.sum(pred2 == task['expected']) / task['expected'].size
                print(f"Prediction 2 Accuracy: {accuracy2*100:.1f}%")

    return pred1, pred2


def test_curiosity_signals():
    """Test curiosity signal computations."""
    print("\n" + "="*70)
    print("Testing Curiosity Signals")
    print("="*70)

    from arc_curiosity_solver.curiosity.signals import CuriositySignals

    curiosity = CuriositySignals()

    # Test Bayesian surprise
    prior = {'mean': np.array([0.5, 0.5]), 'cov': np.eye(2) * 0.1}
    posterior = {'mean': np.array([0.8, 0.2]), 'cov': np.eye(2) * 0.05}

    surprise = curiosity.bayesian_surprise(prior, posterior)
    print(f"\n1. Bayesian Surprise: {surprise:.4f}")
    print("   (Measures belief update magnitude)")

    # Test epistemic uncertainty
    predictions = [0.3, 0.7, 0.4, 0.6, 0.5]
    uncertainty = curiosity.epistemic_uncertainty(predictions)
    print(f"\n2. Epistemic Uncertainty: {uncertainty:.4f}")
    print("   (Measures model disagreement)")

    # Test learning progress
    for perf in [0.3, 0.4, 0.5, 0.6, 0.7]:
        curiosity.learning_progress(perf)

    lp = curiosity.learning_progress(0.8)
    print(f"\n3. Learning Progress: {lp:.4f}")
    print("   (Measures recent improvement)")

    # Test information gain
    ig = curiosity.information_gain(1.0, 0.3)
    print(f"\n4. Information Gain: {ig:.4f}")
    print("   (Expected reduction in entropy)")

    # Test empowerment
    action_outcomes = {
        'a1': [1, 2, 3, 4],
        'a2': [1, 1, 1, 1],
        'a3': [5, 6, 7, 8, 9]
    }
    empower = curiosity.empowerment(action_outcomes, method='diversity')
    print(f"\n5. Empowerment: {empower:.4f}")
    print("   (Diversity of action outcomes)")

    # Combined curiosity
    combined = curiosity.combined_curiosity(
        surprise=surprise,
        uncertainty=uncertainty,
        progress=lp,
        info_gain=ig,
        empower=empower
    )
    print(f"\n6. Combined Curiosity: {combined:.4f}")
    print("   (Weighted combination of all signals)")


def test_belief_dynamics():
    """Test belief space dynamics."""
    print("\n" + "="*70)
    print("Testing Belief Dynamics")
    print("="*70)

    from arc_curiosity_solver.belief_dynamics.belief_space import BeliefSpace, Hypothesis

    # Create belief space with hypotheses
    hypotheses = [
        Hypothesis(None, "translate", {}, "spatial"),
        Hypothesis(None, "rotate", {}, "spatial"),
        Hypothesis(None, "reflect", {}, "spatial"),
    ]

    space = BeliefSpace(hypotheses)

    print(f"\nInitial beliefs: {space.beliefs}")
    print(f"Initial entropy: {space.entropy():.4f}")

    # Simulate evidence updates
    def likelihood_fn(h, evidence):
        # Simulate: 'rotate' gets higher likelihood
        if h.name == "rotate":
            return 0.8
        elif h.name == "translate":
            return 0.3
        else:
            return 0.1

    print("\nUpdating beliefs with evidence...")
    space.bayesian_update("evidence_1", likelihood_fn)

    print(f"Updated beliefs: {space.beliefs}")
    print(f"Updated entropy: {space.entropy():.4f}")
    print(f"KL divergence: {space.kl_divergence():.4f}")

    # Get top hypothesis
    top = space.top_k_hypotheses(k=1)
    print(f"\nTop hypothesis: {top[0][0].name} (belief={top[0][1]:.3f})")


def test_active_inference():
    """Test active inference engine."""
    print("\n" + "="*70)
    print("Testing Active Inference Engine")
    print("="*70)

    from arc_curiosity_solver.active_inference.engine import ActiveInferenceEngine

    engine = ActiveInferenceEngine(learning_rate=0.1)

    # Initialize with 3 hypotheses
    state = engine.initialize_state(n_hypotheses=3)
    print(f"\nInitial beliefs: {state.beliefs}")

    # Simulate observations and predictions
    observations = [
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, 0]]),
    ]

    predictions = [
        [np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]])],
        [np.array([[0, 1], [0, 0]]), np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])],
    ]

    print("\nActive inference updates:")
    for i, (obs, preds) in enumerate(zip(observations, predictions)):
        new_beliefs = engine.update_beliefs(state.beliefs, obs, preds, method='bayes')
        state.beliefs = new_beliefs
        print(f"  Step {i+1}: beliefs = {new_beliefs}, entropy = {-np.sum(new_beliefs * np.log(new_beliefs + 1e-10)):.3f}")

    metrics = engine.track_convergence()
    print(f"\nConvergence metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def main():
    """Main test function."""

    print("\n" + "="*70)
    print("ARC CURIOSITY-DRIVEN ACTIVE INFERENCE SOLVER")
    print("Comprehensive System Test")
    print("="*70)

    # Test individual components
    print("\n" + "="*70)
    print("PART 1: Component Tests")
    print("="*70)

    test_curiosity_signals()
    test_belief_dynamics()
    test_active_inference()

    # Test full solver
    print("\n\n" + "="*70)
    print("PART 2: Full Solver Tests")
    print("="*70)

    # Create solver
    solver = ARCCuriositySolver(
        workspace_capacity=7,
        learning_rate=0.1,
        exploration_bonus=1.0,
        n_hypotheses_to_explore=30
    )

    # Create test tasks
    tasks = create_test_tasks()

    results = []
    for task in tasks:
        try:
            pred1, pred2 = test_solver_on_task(solver, task, verbose=True)
            results.append({
                'task': task['name'],
                'success': True,
                'pred1_shape': pred1.shape,
                'pred2_shape': pred2.shape
            })
        except Exception as e:
            print(f"\nERROR testing {task['name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'task': task['name'],
                'success': False,
                'error': str(e)
            })

    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for result in results:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"{status} - {result['task']}")

    successes = sum(1 for r in results if r['success'])
    print(f"\nTotal: {successes}/{len(results)} tasks passed")

    # Show solver statistics
    print("\n" + "="*70)
    print("SOLVER STATISTICS")
    print("="*70)

    state = solver.get_solver_state()
    print(f"\nTotal solve attempts: {state['solve_attempts']}")
    print(f"Belief entropy: {state['belief_entropy']:.4f}")

    print("\n" + "="*70)
    print("All tests complete!")
    print("="*70)


if __name__ == "__main__":
    main()
