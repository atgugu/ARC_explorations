"""
Pattern-Based ARC Curiosity Solver

Extends the base solver with smart pattern inference:
1. Analyzes training examples to extract invariant patterns
2. Generates hypotheses that match observed patterns
3. Falls back to generic hypotheses if no patterns found

Safe approach: Only generates transformations matching detected invariants.
"""

import numpy as np
from typing import List, Tuple, Optional
from .solver import ARCCuriositySolver
from .belief_dynamics.belief_space import Hypothesis
from .core.pattern_inference import PatternAnalyzer, PatternBasedHypothesisGenerator, InvariantPattern
from .transformations.arc_primitives import ARCPrimitives


class PatternBasedARCCuriositySolver(ARCCuriositySolver):
    """
    Enhanced solver with pattern inference.

    Key improvement:
    - Analyzes training examples to find invariant patterns
    - Only generates hypotheses matching observed patterns
    - Falls back to generic if no patterns detected
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern_analyzer = PatternAnalyzer()
        self.pattern_generator = PatternBasedHypothesisGenerator()
        self.detected_patterns: List[InvariantPattern] = []
        self.used_pattern_inference = False

    def _generate_hypotheses(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                            test_input: np.ndarray) -> List[Hypothesis]:
        """
        Generate hypotheses using pattern inference.

        Safe approach:
        1. Analyze training pairs for invariant patterns
        2. Generate pattern-based hypotheses (high confidence)
        3. Add some generic hypotheses for diversity
        4. Fall back to base if no patterns found
        """
        hypotheses = []

        # Step 1: Analyze training pairs to extract patterns
        self.detected_patterns = self.pattern_analyzer.analyze_training_pairs(train_pairs)

        if self.detected_patterns:
            self.used_pattern_inference = True

            if self.verbose:
                print(f"  🔍 Detected {len(self.detected_patterns)} invariant patterns:")
                for i, pattern in enumerate(self.detected_patterns[:5], 1):  # Show top 5
                    print(f"     {i}. {pattern}")

            # Step 2: Generate hypotheses from patterns
            pattern_hypotheses = self.pattern_generator.generate_from_patterns(
                self.detected_patterns, test_input
            )

            # Convert to Hypothesis objects
            for transform_obj, description, confidence in pattern_hypotheses:
                hyp = Hypothesis(
                    program=transform_obj,
                    name=f"pattern_{len(hypotheses)}",
                    parameters={'description': description},
                    activation=confidence
                )
                hypotheses.append(hyp)

            # Step 3: Add some diversity with generic hypotheses (lower initial belief)
            max_generic = 3  # Add a few generic hypotheses for diversity
            if len(hypotheses) < max_generic:
                generic_hyps = super()._generate_hypotheses(train_pairs, test_input)
                # Lower their initial belief to prioritize pattern-based ones
                for hyp in generic_hyps[:max_generic]:
                    hyp.activation *= 0.3  # Lower activation for generic
                    hypotheses.append(hyp)

        else:
            # No patterns detected - fall back to base solver
            if self.verbose:
                print("  ⚠️  No invariant patterns detected, using generic hypotheses")

            hypotheses = super()._generate_hypotheses(train_pairs, test_input)

        return hypotheses

    def solve(self, train_pairs: List[Tuple], test_input: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve with pattern inference tracking.
        """
        self.verbose = verbose
        self.used_pattern_inference = False
        self.detected_patterns = []

        # Call parent solve
        result = super().solve(train_pairs, test_input, verbose)

        if verbose and self.used_pattern_inference:
            print(f"\n  ✅ Used pattern inference with {len(self.detected_patterns)} patterns")

        return result

    def get_pattern_info(self) -> dict:
        """Get information about detected patterns."""
        return {
            'used_pattern_inference': self.used_pattern_inference,
            'num_patterns': len(self.detected_patterns),
            'patterns': [
                {
                    'type': p.transform_type,
                    'description': f"{p.condition_desc} → {p.action_desc}",
                    'confidence': p.confidence,
                    'support': p.support_count
                }
                for p in self.detected_patterns
            ]
        }
