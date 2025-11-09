"""
Curiosity Signals for ARC Reasoning

Implements:
- Bayesian Surprise
- Epistemic Uncertainty
- Learning Progress
- Information Gain
- Empowerment
"""

import numpy as np
from scipy.stats import entropy
from scipy.special import kl_div
from typing import Dict, List, Any, Tuple
import warnings


class CuriositySignals:
    """Compute various curiosity signals for exploration guidance."""

    def __init__(self):
        self.history = {
            'performance': [],
            'timestamps': [],
            'contexts': []
        }

    def bayesian_surprise(self,
                         prior_params: Dict[str, np.ndarray],
                         posterior_params: Dict[str, np.ndarray]) -> float:
        """
        Compute Bayesian surprise as KL divergence between posterior and prior.

        Surprise_M(e) = KL[p(θ|D∪{e}) || p(θ|D)]

        Args:
            prior_params: Parameters of prior distribution (mean, cov)
            posterior_params: Parameters of posterior distribution (mean, cov)

        Returns:
            KL divergence (surprise) value
        """
        # For Gaussian distributions
        prior_mean = prior_params.get('mean', np.zeros(1))
        prior_cov = prior_params.get('cov', np.eye(len(prior_mean)))
        posterior_mean = posterior_params.get('mean', np.zeros(1))
        posterior_cov = posterior_params.get('cov', np.eye(len(posterior_mean)))

        # Ensure same dimensionality
        if len(prior_mean) != len(posterior_mean):
            return 0.0

        try:
            # KL divergence for Gaussian distributions
            d = len(prior_mean)

            # Add small regularization for numerical stability
            prior_cov = prior_cov + 1e-6 * np.eye(d)
            posterior_cov = posterior_cov + 1e-6 * np.eye(d)

            inv_posterior_cov = np.linalg.inv(posterior_cov)

            log_det_ratio = np.log(np.linalg.det(posterior_cov) / np.linalg.det(prior_cov))
            trace_term = np.trace(inv_posterior_cov @ prior_cov)

            mean_diff = posterior_mean - prior_mean
            mahalanobis = mean_diff.T @ inv_posterior_cov @ mean_diff

            kl = 0.5 * (log_det_ratio + trace_term + mahalanobis - d)

            return max(0.0, float(kl))

        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f"Error computing Bayesian surprise: {e}")
            return 0.0

    def epistemic_uncertainty(self,
                             predictions: List[float],
                             method: str = 'variance') -> float:
        """
        Compute epistemic uncertainty from ensemble predictions.

        Var_epistemic[score(h)]

        Args:
            predictions: List of predictions from ensemble models
            method: 'variance', 'entropy', or 'disagreement'

        Returns:
            Uncertainty value
        """
        if len(predictions) < 2:
            return 0.0

        predictions = np.array(predictions)

        if method == 'variance':
            return float(np.var(predictions))

        elif method == 'entropy':
            # Treat predictions as probabilities
            predictions = np.clip(predictions, 1e-10, 1.0)
            predictions = predictions / predictions.sum()
            return float(entropy(predictions))

        elif method == 'disagreement':
            # Pairwise disagreement
            n = len(predictions)
            disagreement = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    disagreement += abs(predictions[i] - predictions[j])
            return disagreement / (n * (n-1) / 2) if n > 1 else 0.0

        return float(np.var(predictions))

    def learning_progress(self,
                         current_performance: float,
                         time_window: int = 5) -> float:
        """
        Compute learning progress as recent performance improvement.

        LP(t) = m(t) - m(t-Δ)

        Args:
            current_performance: Current performance metric
            time_window: Number of past steps to compare

        Returns:
            Learning progress value
        """
        self.history['performance'].append(current_performance)

        if len(self.history['performance']) < time_window + 1:
            return 0.0

        recent = self.history['performance'][-time_window-1:-1]
        avg_past = np.mean(recent)

        return float(current_performance - avg_past)

    def information_gain(self,
                        prior_entropy: float,
                        expected_posterior_entropy: float) -> float:
        """
        Compute expected information gain.

        IG = H[X] - E_o[H[X|o]]

        Args:
            prior_entropy: Entropy before observation
            expected_posterior_entropy: Expected entropy after observation

        Returns:
            Information gain value
        """
        return max(0.0, prior_entropy - expected_posterior_entropy)

    def empowerment(self,
                   action_outcomes: Dict[Any, List[Any]],
                   method: str = 'mutual_info') -> float:
        """
        Compute empowerment as mutual information between actions and outcomes.

        Empower(s) ≈ I(A; S' | S=s)

        Args:
            action_outcomes: Dict mapping actions to possible outcomes
            method: 'mutual_info' or 'diversity'

        Returns:
            Empowerment value
        """
        if not action_outcomes:
            return 0.0

        if method == 'diversity':
            # Simple diversity measure: unique outcomes per action
            diversities = []
            for action, outcomes in action_outcomes.items():
                unique_outcomes = len(set(tuple(o) if isinstance(o, (list, np.ndarray))
                                            else o for o in outcomes))
                diversities.append(unique_outcomes)
            return float(np.mean(diversities)) if diversities else 0.0

        elif method == 'mutual_info':
            # Approximate mutual information
            n_actions = len(action_outcomes)
            if n_actions == 0:
                return 0.0

            # Compute entropies
            total_outcomes = []
            for outcomes in action_outcomes.values():
                total_outcomes.extend(outcomes)

            # H(S') - H(S'|A)
            outcome_entropy = self._compute_outcome_entropy(total_outcomes)
            conditional_entropy = 0.0

            for action, outcomes in action_outcomes.items():
                action_prob = 1.0 / n_actions
                action_entropy = self._compute_outcome_entropy(outcomes)
                conditional_entropy += action_prob * action_entropy

            mi = outcome_entropy - conditional_entropy
            return max(0.0, float(mi))

        return 0.0

    def _compute_outcome_entropy(self, outcomes: List[Any]) -> float:
        """Compute entropy of outcome distribution."""
        if not outcomes:
            return 0.0

        # Convert outcomes to hashable types for counting
        hashable_outcomes = []
        for o in outcomes:
            if isinstance(o, np.ndarray):
                hashable_outcomes.append(tuple(o.flatten()))
            elif isinstance(o, list):
                hashable_outcomes.append(tuple(o))
            else:
                hashable_outcomes.append(o)

        # Count frequencies
        unique, counts = np.unique(hashable_outcomes, return_counts=True)
        probs = counts / len(hashable_outcomes)

        return float(entropy(probs))

    def combined_curiosity(self,
                          surprise: float = 0.0,
                          uncertainty: float = 0.0,
                          progress: float = 0.0,
                          info_gain: float = 0.0,
                          empower: float = 0.0,
                          weights: Dict[str, float] = None) -> float:
        """
        Combine multiple curiosity signals with weights.

        Args:
            surprise: Bayesian surprise value
            uncertainty: Epistemic uncertainty value
            progress: Learning progress value
            info_gain: Information gain value
            empower: Empowerment value
            weights: Dictionary of weights for each signal

        Returns:
            Combined curiosity score
        """
        if weights is None:
            weights = {
                'surprise': 0.3,
                'uncertainty': 0.25,
                'progress': 0.2,
                'info_gain': 0.15,
                'empower': 0.1
            }

        curiosity = (
            weights.get('surprise', 0.0) * surprise +
            weights.get('uncertainty', 0.0) * uncertainty +
            weights.get('progress', 0.0) * progress +
            weights.get('info_gain', 0.0) * info_gain +
            weights.get('empower', 0.0) * empower
        )

        return float(curiosity)


class TaskCuriosityScorer:
    """Compute curiosity scores for tasks."""

    def __init__(self, signals: CuriositySignals):
        self.signals = signals

    def score_task(self,
                   task: Any,
                   solver_state: Dict,
                   alpha: float = 1.0,
                   beta: float = 0.8,
                   gamma: float = 0.6,
                   delta: float = 0.3) -> float:
        """
        Compute task curiosity score.

        C_task(τ) = α·IG_solver(τ) + β·Surprise_prior(τ) + γ·LP_forecast(τ) - δ·Redundancy(τ)

        Args:
            task: ARC task
            solver_state: Current solver state
            alpha, beta, gamma, delta: Weight parameters

        Returns:
            Task curiosity score
        """
        # Extract metrics from solver state
        ig = solver_state.get('information_gain', 0.0)
        surprise = solver_state.get('surprise', 0.0)
        lp_forecast = solver_state.get('learning_progress_forecast', 0.0)
        redundancy = solver_state.get('redundancy', 0.0)

        score = (alpha * ig +
                beta * surprise +
                gamma * lp_forecast -
                delta * redundancy)

        return float(score)


class HypothesisCuriosityScorer:
    """Compute curiosity scores for hypotheses."""

    def __init__(self, signals: CuriositySignals):
        self.signals = signals

    def score_hypothesis(self,
                        hypothesis: Any,
                        fit_predictions: List[float],
                        info_gain: float = 0.0,
                        empowerment: float = 0.0,
                        alpha: float = 1.0,
                        beta: float = 0.8,
                        rho: float = 0.5) -> float:
        """
        Compute hypothesis curiosity score.

        Curiosity(h) = α·Var_epistemic[Fit(h)] + β·IG(h) + ρ·Empower(h)

        Args:
            hypothesis: Candidate hypothesis
            fit_predictions: Ensemble predictions of fit
            info_gain: Expected information gain
            empowerment: Empowerment value
            alpha, beta, rho: Weight parameters

        Returns:
            Hypothesis curiosity score
        """
        epistemic_var = self.signals.epistemic_uncertainty(fit_predictions)

        score = (alpha * epistemic_var +
                beta * info_gain +
                rho * empowerment)

        return float(score)
