"""
Belief Dynamics Over Program Space

Implements continuous dynamics in probability space over transformation hypotheses.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Callable, Optional
from dataclasses import dataclass
from scipy.special import softmax
import warnings


@dataclass
class Hypothesis:
    """Represents a transformation hypothesis."""
    program: Any  # The actual program/transformation
    name: str
    parameters: Dict[str, Any]
    activation: float = 0.0  # Belief strength
    evidence_count: int = 0
    success_count: int = 0


class BeliefSpace:
    """
    Manages belief dynamics over hypothesis space.

    Implements:
    - Bayesian belief updating
    - Continuous belief flow
    - Information geometry
    """

    def __init__(self, hypotheses: List[Hypothesis] = None):
        self.hypotheses = hypotheses or []
        self.beliefs = np.ones(len(self.hypotheses)) / max(1, len(self.hypotheses))
        self.prior_beliefs = self.beliefs.copy()
        self.time = 0.0
        self.trajectory = []

    def add_hypothesis(self, hypothesis: Hypothesis):
        """Add a new hypothesis to the space."""
        self.hypotheses.append(hypothesis)
        # Redistribute beliefs uniformly
        self.beliefs = np.ones(len(self.hypotheses)) / len(self.hypotheses)
        self.prior_beliefs = self.beliefs.copy()

    def bayesian_update(self,
                       evidence: Any,
                       likelihood_fn: Callable[[Hypothesis, Any], float],
                       normalize: bool = True):
        """
        Perform Bayesian update: P(h|D_{t+1}) = P(e|h) * P(h|D_t) / P(e|D_t)

        Args:
            evidence: New observation/evidence
            likelihood_fn: Function that computes P(e|h) for each hypothesis
            normalize: Whether to normalize to probability distribution
        """
        if len(self.hypotheses) == 0:
            return

        # Compute likelihoods for each hypothesis
        likelihoods = np.array([
            likelihood_fn(h, evidence) for h in self.hypotheses
        ])

        # Add small epsilon to avoid numerical issues
        likelihoods = np.clip(likelihoods, 1e-10, None)

        # Bayesian update
        self.prior_beliefs = self.beliefs.copy()
        self.beliefs = self.beliefs * likelihoods

        if normalize and self.beliefs.sum() > 0:
            self.beliefs = self.beliefs / self.beliefs.sum()

        # Update hypothesis activations
        for i, h in enumerate(self.hypotheses):
            h.activation = self.beliefs[i]
            h.evidence_count += 1

    def continuous_flow_update(self,
                              evidence: Any,
                              likelihood_fn: Callable[[Hypothesis, Any], float],
                              dt: float = 0.1):
        """
        Update beliefs via continuous flow dynamics.

        dP(h)/dt = P(h) · [log P(e|h) - ⟨log P(e|h')⟩]

        Args:
            evidence: New observation
            likelihood_fn: Likelihood function
            dt: Time step
        """
        if len(self.hypotheses) == 0:
            return

        # Compute log-likelihoods
        log_likelihoods = np.array([
            np.log(np.clip(likelihood_fn(h, evidence), 1e-10, None))
            for h in self.hypotheses
        ])

        # Expected log-likelihood
        expected_log_lik = np.dot(self.beliefs, log_likelihoods)

        # Flow dynamics
        dP_dt = self.beliefs * (log_likelihoods - expected_log_lik)

        # Update beliefs
        self.prior_beliefs = self.beliefs.copy()
        self.beliefs = self.beliefs + dt * dP_dt

        # Ensure non-negativity and normalization
        self.beliefs = np.clip(self.beliefs, 0, None)
        if self.beliefs.sum() > 0:
            self.beliefs = self.beliefs / self.beliefs.sum()

        self.time += dt

        # Store trajectory
        self.trajectory.append({
            'time': self.time,
            'beliefs': self.beliefs.copy(),
            'entropy': self.entropy()
        })

        # Update hypothesis activations
        for i, h in enumerate(self.hypotheses):
            h.activation = self.beliefs[i]

    def entropy(self) -> float:
        """Compute entropy of current belief distribution."""
        p = np.clip(self.beliefs, 1e-10, None)
        return float(-np.sum(p * np.log(p)))

    def kl_divergence(self, other_beliefs: np.ndarray = None) -> float:
        """
        Compute KL divergence from prior to current beliefs.

        KL[P(h|D_{t+1}) || P(h|D_t)]

        Args:
            other_beliefs: Beliefs to compare against (default: prior_beliefs)

        Returns:
            KL divergence
        """
        if other_beliefs is None:
            other_beliefs = self.prior_beliefs

        p = np.clip(self.beliefs, 1e-10, None)
        q = np.clip(other_beliefs, 1e-10, None)

        return float(np.sum(p * np.log(p / q)))

    def top_k_hypotheses(self, k: int = 2) -> List[Tuple[Hypothesis, float]]:
        """
        Get top k hypotheses by belief.

        Args:
            k: Number of hypotheses to return

        Returns:
            List of (hypothesis, belief) tuples
        """
        if len(self.hypotheses) == 0:
            return []

        indices = np.argsort(self.beliefs)[-k:][::-1]
        return [(self.hypotheses[i], self.beliefs[i]) for i in indices]

    def expected_information_gain(self,
                                 test_evidence: Any,
                                 likelihood_fn: Callable) -> float:
        """
        Compute expected information gain from observing test evidence.

        IG = H[D_t] - E[H[D_{t+1}]]

        Args:
            test_evidence: Evidence to test
            likelihood_fn: Likelihood function

        Returns:
            Expected information gain
        """
        current_entropy = self.entropy()

        # Simulate update for each hypothesis
        expected_entropy = 0.0
        for i, h in enumerate(self.hypotheses):
            # Probability of this hypothesis being true
            p_h = self.beliefs[i]

            # Simulate update assuming this hypothesis is true
            temp_beliefs = self.beliefs.copy()
            likelihood = likelihood_fn(h, test_evidence)
            temp_beliefs = temp_beliefs * likelihood

            if temp_beliefs.sum() > 0:
                temp_beliefs = temp_beliefs / temp_beliefs.sum()

            # Compute entropy of updated beliefs
            p = np.clip(temp_beliefs, 1e-10, None)
            h_updated = -np.sum(p * np.log(p))

            expected_entropy += p_h * h_updated

        return current_entropy - expected_entropy

    def convergence_metrics(self) -> Dict[str, float]:
        """
        Compute metrics about belief convergence.

        Returns:
            Dictionary with convergence metrics
        """
        max_belief = np.max(self.beliefs) if len(self.beliefs) > 0 else 0.0
        entropy = self.entropy()
        max_entropy = np.log(len(self.hypotheses)) if len(self.hypotheses) > 0 else 0.0

        # Gini coefficient as concentration measure
        sorted_beliefs = np.sort(self.beliefs)
        n = len(sorted_beliefs)
        if n > 0 and sorted_beliefs.sum() > 0:
            cumsum = np.cumsum(sorted_beliefs)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        else:
            gini = 0.0

        return {
            'max_belief': float(max_belief),
            'entropy': float(entropy),
            'normalized_entropy': float(entropy / max_entropy) if max_entropy > 0 else 0.0,
            'concentration': float(gini),
            'n_hypotheses': len(self.hypotheses)
        }

    def prune_hypotheses(self, threshold: float = 0.01):
        """Remove hypotheses with very low beliefs."""
        if len(self.hypotheses) == 0:
            return

        # Keep only hypotheses above threshold
        keep_indices = self.beliefs >= threshold

        if not np.any(keep_indices):
            # Keep at least the top hypothesis
            keep_indices[np.argmax(self.beliefs)] = True

        self.hypotheses = [h for i, h in enumerate(self.hypotheses) if keep_indices[i]]
        self.beliefs = self.beliefs[keep_indices]

        # Renormalize
        if self.beliefs.sum() > 0:
            self.beliefs = self.beliefs / self.beliefs.sum()


class HierarchicalBeliefSpace:
    """
    Multi-level belief space for hierarchical reasoning.

    Levels:
    1. Meta-patterns (high-level transformation types)
    2. Compositional rules (combinations of primitives)
    3. Primitive operations (atomic transformations)
    """

    def __init__(self):
        self.levels = {
            'meta': BeliefSpace(),
            'compositional': BeliefSpace(),
            'primitive': BeliefSpace()
        }
        self.cross_level_coupling = 0.3

    def add_hypothesis(self, hypothesis: Hypothesis, level: str):
        """Add hypothesis at specific level."""
        if level in self.levels:
            self.levels[level].add_hypothesis(hypothesis)

    def update_all_levels(self,
                         evidence: Any,
                         likelihood_fns: Dict[str, Callable],
                         method: str = 'bayesian'):
        """
        Update beliefs at all levels with cross-level coupling.

        Args:
            evidence: Observation
            likelihood_fns: Dict mapping levels to likelihood functions
            method: 'bayesian' or 'continuous'
        """
        # Bottom-up pass: update from evidence
        for level in ['primitive', 'compositional', 'meta']:
            if level in likelihood_fns:
                space = self.levels[level]

                if method == 'bayesian':
                    space.bayesian_update(evidence, likelihood_fns[level])
                elif method == 'continuous':
                    space.continuous_flow_update(evidence, likelihood_fns[level])

        # Top-down pass: high-level beliefs influence low-level
        self._apply_cross_level_coupling()

    def _apply_cross_level_coupling(self):
        """Apply top-down modulation from higher to lower levels."""
        # Meta -> Compositional
        if (len(self.levels['meta'].hypotheses) > 0 and
            len(self.levels['compositional'].hypotheses) > 0):

            # High-level beliefs modulate mid-level
            meta_top = self.levels['meta'].top_k_hypotheses(k=1)
            if meta_top:
                # Boost compositional hypotheses consistent with top meta-hypothesis
                # (This is simplified; real implementation would use structured relationships)
                self.levels['compositional'].beliefs *= (
                    1.0 + self.cross_level_coupling * meta_top[0][1]
                )
                self.levels['compositional'].beliefs /= self.levels['compositional'].beliefs.sum()

        # Compositional -> Primitive
        if (len(self.levels['compositional'].hypotheses) > 0 and
            len(self.levels['primitive'].hypotheses) > 0):

            comp_top = self.levels['compositional'].top_k_hypotheses(k=1)
            if comp_top:
                self.levels['primitive'].beliefs *= (
                    1.0 + self.cross_level_coupling * comp_top[0][1]
                )
                self.levels['primitive'].beliefs /= self.levels['primitive'].beliefs.sum()

    def get_top_hypotheses(self, k: int = 2) -> Dict[str, List[Tuple[Hypothesis, float]]]:
        """Get top k hypotheses from all levels."""
        return {
            level: space.top_k_hypotheses(k)
            for level, space in self.levels.items()
        }

    def overall_entropy(self) -> float:
        """Compute weighted average entropy across levels."""
        entropies = [space.entropy() for space in self.levels.values()]
        weights = [len(space.hypotheses) for space in self.levels.values()]
        total_weight = sum(weights)

        if total_weight == 0:
            return 0.0

        return sum(e * w for e, w in zip(entropies, weights)) / total_weight
