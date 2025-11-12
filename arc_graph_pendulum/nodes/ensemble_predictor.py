"""
Ensemble Predictor - Combines multiple programs using weighted voting.

Key insight from V7 negative result:
- Single-program selection often picks slightly wrong program
- Top-K programs often contain the correct one
- Ensemble voting can combine their strengths

Strategy:
1. Generate top-K programs (K=5) instead of selecting best
2. Weight by training performance
3. Pixel-wise weighted majority voting
"""

import numpy as np
from typing import List, Dict, Any, Callable, Tuple
from collections import Counter


class EnsemblePredictor:
    """Combines multiple programs using weighted voting."""

    def __init__(self, top_k: int = 5, confidence_threshold: float = 0.7):
        """
        Initialize ensemble predictor.

        Args:
            top_k: Number of top programs to use in ensemble
            confidence_threshold: Minimum score to include program
        """
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

    def select_top_k_programs(self, programs: List[Dict[str, Any]],
                              train_examples: List[Tuple[np.ndarray, np.ndarray]],
                              verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Select top-K programs based on training performance.

        Args:
            programs: List of program dictionaries
            train_examples: Training examples for evaluation
            verbose: Print debug info

        Returns:
            Top-K programs with their scores
        """
        if not programs:
            return []

        # Evaluate all programs on training
        scored_programs = []

        for prog in programs:
            score = self._evaluate_program(prog['function'], train_examples)

            # Only include if above confidence threshold
            if score >= self.confidence_threshold:
                scored_programs.append({
                    'function': prog['function'],
                    'type': prog.get('type', 'unknown'),
                    'confidence': prog.get('confidence', 0.9),
                    'train_score': score,
                    'description': prog.get('description', '')
                })

        if not scored_programs:
            # If no programs meet threshold, take the best one anyway
            if verbose:
                print(f"  Warning: No programs above threshold {self.confidence_threshold}")
            best_prog = max(programs, key=lambda p: self._evaluate_program(p['function'], train_examples))
            scored_programs = [{
                'function': best_prog['function'],
                'type': best_prog.get('type', 'unknown'),
                'confidence': best_prog.get('confidence', 0.9),
                'train_score': self._evaluate_program(best_prog['function'], train_examples),
                'description': best_prog.get('description', '')
            }]

        # Sort by training score and take top-K
        scored_programs.sort(key=lambda p: p['train_score'], reverse=True)
        top_k = scored_programs[:self.top_k]

        if verbose:
            print(f"\n  Selected {len(top_k)} programs for ensemble:")
            for i, prog in enumerate(top_k, 1):
                print(f"    {i}. {prog['type']} (score={prog['train_score']:.3f})")

        return top_k

    def _evaluate_program(self, program_func: Callable,
                         train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Evaluate program on training examples.

        Args:
            program_func: Program function to evaluate
            train_examples: Training input-output pairs

        Returns:
            Average IoU score
        """
        if not train_examples:
            return 0.0

        scores = []

        for input_grid, output_grid in train_examples:
            try:
                predicted = program_func(input_grid)

                # Compute IoU
                if predicted.shape == output_grid.shape:
                    iou = np.sum(predicted == output_grid) / predicted.size
                else:
                    iou = 0.0

                scores.append(iou)

            except Exception:
                scores.append(0.0)

        return np.mean(scores) if scores else 0.0

    def predict_with_ensemble(self, programs: List[Dict[str, Any]],
                             input_grid: np.ndarray,
                             verbose: bool = False) -> np.ndarray:
        """
        Generate prediction using ensemble of programs.

        Args:
            programs: Top-K programs with scores
            input_grid: Test input
            verbose: Print debug info

        Returns:
            Ensemble prediction
        """
        if not programs:
            return input_grid.copy()

        # If only one program, just use it
        if len(programs) == 1:
            return programs[0]['function'](input_grid)

        # Get predictions from all programs
        predictions = []
        weights = []
        valid_programs = []

        for prog in programs:
            try:
                pred = prog['function'](input_grid)
                predictions.append(pred)
                weights.append(prog['train_score'])
                valid_programs.append(prog)

            except Exception as e:
                if verbose:
                    print(f"  Warning: Program {prog['type']} failed: {e}")
                continue

        if not predictions:
            return input_grid.copy()

        # If only one valid prediction, use it
        if len(predictions) == 1:
            return predictions[0]

        # Check if all predictions have the same shape
        shapes = [p.shape for p in predictions]
        if len(set(shapes)) > 1:
            # Different shapes - use highest-scoring program
            if verbose:
                print(f"  Shape mismatch in ensemble, using best program")
            best_idx = np.argmax(weights)
            return predictions[best_idx]

        # All same shape - do weighted voting
        result = self._weighted_vote(predictions, weights, verbose=verbose)

        if verbose:
            # Show agreement statistics
            agreements = [np.sum(pred == result) / pred.size for pred in predictions]
            print(f"  Ensemble agreement: {np.mean(agreements):.3f} avg")

        return result

    def _weighted_vote(self, predictions: List[np.ndarray],
                       weights: List[float],
                       verbose: bool = False) -> np.ndarray:
        """
        Perform weighted pixel-wise voting.

        Args:
            predictions: List of prediction grids (all same shape)
            weights: Weights for each prediction (training scores)
            verbose: Print debug info

        Returns:
            Voted prediction grid
        """
        # Stack predictions: shape (num_programs, height, width)
        stacked = np.stack(predictions, axis=0)
        weights_array = np.array(weights)

        # Normalize weights to sum to 1
        weights_array = weights_array / np.sum(weights_array)

        height, width = predictions[0].shape
        result = np.zeros((height, width), dtype=np.int32)

        # For each pixel position
        for i in range(height):
            for j in range(width):
                # Get all predictions for this pixel
                pixel_values = stacked[:, i, j]

                # Weight votes by program scores
                vote_counts = {}
                for value, weight in zip(pixel_values, weights_array):
                    value = int(value)
                    if value not in vote_counts:
                        vote_counts[value] = 0.0
                    vote_counts[value] += weight

                # Select value with highest weighted vote
                result[i, j] = max(vote_counts.items(), key=lambda x: x[1])[0]

        return result

    def predict_multiple_test_examples(self, programs: List[Dict[str, Any]],
                                       test_inputs: List[np.ndarray],
                                       verbose: bool = False) -> List[np.ndarray]:
        """
        Generate predictions for multiple test examples.

        Args:
            programs: Top-K programs with scores
            test_inputs: List of test input grids
            verbose: Print debug info

        Returns:
            List of ensemble predictions
        """
        predictions = []

        for i, test_input in enumerate(test_inputs):
            if verbose:
                print(f"\n  Ensemble prediction for test {i+1}:")

            pred = self.predict_with_ensemble(programs, test_input, verbose=verbose)
            predictions.append(pred)

        return predictions

    def get_ensemble_confidence(self, programs: List[Dict[str, Any]]) -> float:
        """
        Get confidence score for ensemble.

        Higher confidence when:
        - Multiple strong programs (high scores)
        - Programs have diverse approaches
        - Top programs agree on training

        Args:
            programs: Top-K programs with scores

        Returns:
            Confidence score (0-1)
        """
        if not programs:
            return 0.0

        if len(programs) == 1:
            return programs[0]['train_score']

        # Average of top programs
        avg_score = np.mean([p['train_score'] for p in programs])

        # Bonus for diversity (different types)
        types = [p['type'] for p in programs]
        diversity = len(set(types)) / len(types)

        # Bonus for multiple strong programs
        strong_count = sum(1 for p in programs if p['train_score'] >= 0.95)
        strength_bonus = min(strong_count / len(programs), 0.2)

        confidence = avg_score * 0.7 + diversity * 0.2 + strength_bonus

        return min(confidence, 1.0)
