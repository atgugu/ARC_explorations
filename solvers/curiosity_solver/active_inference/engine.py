"""
Active Inference Engine for ARC

Implements active inference principles:
- Free energy minimization
- Predictive coding
- Active sampling
- Belief updating during inference
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Callable, Optional
from dataclasses import dataclass, field
import warnings


@dataclass
class InferenceState:
    """State of the active inference process."""
    beliefs: np.ndarray
    prediction_errors: List[float] = field(default_factory=list)
    free_energy: List[float] = field(default_factory=list)
    surprises: List[float] = field(default_factory=list)
    time_steps: int = 0


class ActiveInferenceEngine:
    """
    Active inference engine for continuous learning during inference.

    Key principles:
    1. Minimize free energy (surprise + uncertainty)
    2. Update beliefs based on prediction errors
    3. Select actions that reduce uncertainty
    4. Balance exploration and exploitation
    """

    def __init__(self,
                 learning_rate: float = 0.1,
                 precision: float = 1.0,
                 temperature: float = 1.0):
        """
        Initialize active inference engine.

        Args:
            learning_rate: Rate of belief updating
            precision: Inverse variance of prediction errors (confidence)
            temperature: Exploration temperature
        """
        self.learning_rate = learning_rate
        self.precision = precision
        self.temperature = temperature
        self.state = None

    def initialize_state(self, n_hypotheses: int) -> InferenceState:
        """Initialize inference state with uniform beliefs."""
        beliefs = np.ones(n_hypotheses) / n_hypotheses
        self.state = InferenceState(beliefs=beliefs)
        return self.state

    def compute_free_energy(self,
                           beliefs: np.ndarray,
                           observations: List[Any],
                           generative_model: Callable) -> float:
        """
        Compute variational free energy.

        F = -log P(observations) + KL[Q(h) || P(h)]
          ≈ prediction_error + complexity

        Args:
            beliefs: Current belief distribution Q(h)
            observations: Observed data
            generative_model: Function that generates predictions

        Returns:
            Free energy value
        """
        # Prediction error term (accuracy)
        prediction_error = 0.0
        for obs in observations:
            # Generate predictions from each hypothesis
            predictions = generative_model(beliefs, obs)

            # Compute weighted prediction error
            error = self._compute_prediction_error(obs, predictions, beliefs)
            prediction_error += error

        # Complexity term (KL divergence from prior)
        prior = np.ones_like(beliefs) / len(beliefs)
        complexity = self._kl_divergence(beliefs, prior)

        free_energy = prediction_error + complexity
        return float(free_energy)

    def _compute_prediction_error(self,
                                 observation: Any,
                                 predictions: List[Any],
                                 beliefs: np.ndarray) -> float:
        """Compute precision-weighted prediction error."""
        # Simple L2 error for now
        if isinstance(observation, np.ndarray):
            errors = []
            for pred in predictions:
                if isinstance(pred, np.ndarray):
                    error = np.sum((observation - pred) ** 2)
                    errors.append(error)

            if errors:
                weighted_error = np.dot(beliefs, errors)
                return self.precision * weighted_error

        return 0.0

    def _kl_divergence(self, q: np.ndarray, p: np.ndarray) -> float:
        """Compute KL divergence KL[q || p]."""
        q = np.clip(q, 1e-10, None)
        p = np.clip(p, 1e-10, None)
        return float(np.sum(q * np.log(q / p)))

    def update_beliefs(self,
                      current_beliefs: np.ndarray,
                      observation: Any,
                      predictions: List[Any],
                      method: str = 'gradient') -> np.ndarray:
        """
        Update beliefs based on prediction errors.

        Args:
            current_beliefs: Current Q(h)
            observation: New observation
            predictions: Predictions from each hypothesis
            method: 'gradient' or 'bayes'

        Returns:
            Updated beliefs
        """
        if method == 'gradient':
            # Gradient descent on free energy
            # Compute gradients (simplified)
            gradients = self._compute_belief_gradients(
                current_beliefs, observation, predictions
            )

            new_beliefs = current_beliefs - self.learning_rate * gradients

            # Project onto probability simplex
            new_beliefs = np.clip(new_beliefs, 0, None)
            if new_beliefs.sum() > 0:
                new_beliefs = new_beliefs / new_beliefs.sum()

        elif method == 'bayes':
            # Bayesian update based on prediction accuracy
            likelihoods = self._compute_likelihoods(observation, predictions)
            new_beliefs = current_beliefs * likelihoods

            if new_beliefs.sum() > 0:
                new_beliefs = new_beliefs / new_beliefs.sum()

        else:
            new_beliefs = current_beliefs

        # Track prediction error
        if self.state is not None:
            error = self._compute_prediction_error(observation, predictions, current_beliefs)
            self.state.prediction_errors.append(error)
            self.state.time_steps += 1

        return new_beliefs

    def _compute_belief_gradients(self,
                                 beliefs: np.ndarray,
                                 observation: Any,
                                 predictions: List[Any]) -> np.ndarray:
        """Compute gradients of free energy w.r.t. beliefs."""
        n = len(beliefs)
        gradients = np.zeros(n)

        # Numerical gradient estimation
        epsilon = 1e-5
        for i in range(n):
            # Perturb belief i
            beliefs_plus = beliefs.copy()
            beliefs_plus[i] += epsilon
            beliefs_plus = beliefs_plus / beliefs_plus.sum()

            beliefs_minus = beliefs.copy()
            beliefs_minus[i] = max(epsilon, beliefs_minus[i] - epsilon)
            beliefs_minus = beliefs_minus / beliefs_minus.sum()

            # Compute finite difference
            error_plus = self._compute_prediction_error(observation, predictions, beliefs_plus)
            error_minus = self._compute_prediction_error(observation, predictions, beliefs_minus)

            gradients[i] = (error_plus - error_minus) / (2 * epsilon)

        return gradients

    def _compute_likelihoods(self,
                            observation: Any,
                            predictions: List[Any]) -> np.ndarray:
        """Compute likelihood P(observation | hypothesis)."""
        likelihoods = []

        for pred in predictions:
            if isinstance(observation, np.ndarray) and isinstance(pred, np.ndarray):
                # Gaussian likelihood
                error = np.sum((observation - pred) ** 2)
                likelihood = np.exp(-self.precision * error / 2)
                likelihoods.append(likelihood)
            else:
                # Default uniform likelihood
                likelihoods.append(1.0)

        likelihoods = np.array(likelihoods)
        return likelihoods + 1e-10  # Avoid zeros

    def select_informative_action(self,
                                 beliefs: np.ndarray,
                                 possible_actions: List[Any],
                                 expected_info_gains: List[float]) -> Any:
        """
        Select action that maximizes expected information gain.

        This implements active inference: choose actions that resolve uncertainty.

        Args:
            beliefs: Current beliefs
            possible_actions: Available actions
            expected_info_gains: Expected IG for each action

        Returns:
            Selected action
        """
        if not possible_actions:
            return None

        # Softmax selection with temperature
        scores = np.array(expected_info_gains) / self.temperature
        probs = np.exp(scores - np.max(scores))
        probs = probs / probs.sum()

        # Sample action
        action_idx = np.random.choice(len(possible_actions), p=probs)
        return possible_actions[action_idx]

    def compute_expected_free_energy(self,
                                    beliefs: np.ndarray,
                                    action: Any,
                                    outcome_model: Callable) -> float:
        """
        Compute expected free energy for taking an action.

        G = E[surprise] + E[uncertainty]

        Lower G means better action (more informative, less surprising).

        Args:
            beliefs: Current beliefs
            action: Candidate action
            outcome_model: Function predicting outcomes

        Returns:
            Expected free energy
        """
        # Predict outcomes for this action
        predicted_outcomes = outcome_model(action, beliefs)

        # Expected surprise (how surprising will outcomes be)
        expected_surprise = 0.0
        for outcome, prob in predicted_outcomes:
            # Surprise = -log P(outcome | beliefs)
            surprise = -np.log(np.clip(prob, 1e-10, None))
            expected_surprise += prob * surprise

        # Expected uncertainty (how much uncertainty will remain)
        # This is approximated by entropy of predicted belief distribution
        expected_uncertainty = self._compute_expected_entropy(
            beliefs, predicted_outcomes
        )

        return expected_surprise + expected_uncertainty

    def _compute_expected_entropy(self,
                                 beliefs: np.ndarray,
                                 outcomes: List[Tuple[Any, float]]) -> float:
        """Compute expected entropy after observing outcomes."""
        expected_H = 0.0

        for outcome, prob in outcomes:
            # Simulate belief update for this outcome
            # (Simplified: assume uniform impact)
            updated_beliefs = beliefs.copy()
            updated_beliefs = updated_beliefs / updated_beliefs.sum()

            # Entropy of updated beliefs
            p = np.clip(updated_beliefs, 1e-10, None)
            H = -np.sum(p * np.log(p))

            expected_H += prob * H

        return expected_H

    def track_convergence(self) -> Dict[str, Any]:
        """
        Track convergence metrics during inference.

        Returns:
            Dictionary of convergence metrics
        """
        if self.state is None:
            return {}

        metrics = {
            'time_steps': self.state.time_steps,
            'total_prediction_error': sum(self.state.prediction_errors),
            'mean_prediction_error': np.mean(self.state.prediction_errors) if self.state.prediction_errors else 0,
            'belief_entropy': float(-np.sum(
                self.state.beliefs * np.log(np.clip(self.state.beliefs, 1e-10, None))
            )),
            'max_belief': float(np.max(self.state.beliefs)),
            'converged': float(np.max(self.state.beliefs)) > 0.8
        }

        return metrics


class PredictiveCoder:
    """
    Predictive coding mechanism for hierarchical inference.

    Implements:
    - Top-down predictions
    - Bottom-up prediction errors
    - Error-driven learning
    """

    def __init__(self, n_levels: int = 3):
        self.n_levels = n_levels
        self.predictions = [None] * n_levels
        self.errors = [None] * n_levels

    def forward_pass(self,
                    input_data: Any,
                    generative_models: List[Callable]) -> List[Any]:
        """
        Top-down predictive pass.

        Args:
            input_data: Input observation
            generative_models: Models for each level

        Returns:
            Predictions at each level
        """
        # Start from top level
        for level in range(self.n_levels - 1, -1, -1):
            if level == self.n_levels - 1:
                # Top level generates prediction
                self.predictions[level] = generative_models[level](None)
            else:
                # Lower levels predict based on higher level
                self.predictions[level] = generative_models[level](
                    self.predictions[level + 1]
                )

        return self.predictions

    def backward_pass(self,
                     observations: List[Any]) -> List[Any]:
        """
        Bottom-up error propagation.

        Args:
            observations: Observed data at each level

        Returns:
            Prediction errors at each level
        """
        # Compute errors at each level
        for level in range(self.n_levels):
            if level < len(observations) and observations[level] is not None:
                obs = observations[level]
                pred = self.predictions[level]

                if isinstance(obs, np.ndarray) and isinstance(pred, np.ndarray):
                    self.errors[level] = obs - pred
                else:
                    self.errors[level] = None

        return self.errors

    def update_beliefs(self,
                      learning_rate: float = 0.1) -> List[Any]:
        """
        Update internal representations based on errors.

        Args:
            learning_rate: Learning rate

        Returns:
            Updated predictions
        """
        # Update predictions to reduce errors
        for level in range(self.n_levels):
            if self.errors[level] is not None and self.predictions[level] is not None:
                # Simple gradient update
                if isinstance(self.predictions[level], np.ndarray):
                    self.predictions[level] += learning_rate * self.errors[level]

        return self.predictions
