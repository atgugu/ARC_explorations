"""
Stability measurement with Lyapunov-like indicators.
"""

import numpy as np
from typing import List, Dict, Any, Callable
from dataclasses import dataclass


@dataclass
class StabilityMetrics:
    """
    Metrics for measuring trajectory stability.

    Attributes:
        sensitivity: Sensitivity to input perturbations
        variance: Variance across repeated runs
        trajectory_entropy: Entropy of node visitation
        lyapunov_exponent: Approximate Lyapunov exponent
        is_stable: Boolean flag for stability
    """
    sensitivity: float
    variance: float
    trajectory_entropy: float
    lyapunov_exponent: float
    is_stable: bool


class StabilityMeter:
    """
    Measures stability of trajectories using Lyapunov-like probes.
    """

    def __init__(self, num_perturbations: int = 5, epsilon: float = 0.01):
        """
        Initialize stability meter.

        Args:
            num_perturbations: Number of perturbed runs to perform
            epsilon: Perturbation magnitude
        """
        self.num_perturbations = num_perturbations
        self.epsilon = epsilon

    def measure_trajectory_stability(
        self,
        execute_func: Callable,
        input_data: Any,
        metric_func: Callable[[Any, Any], float]
    ) -> StabilityMetrics:
        """
        Measure stability by running with perturbations.

        Args:
            execute_func: Function to execute trajectory
            input_data: Base input data
            metric_func: Function to compute distance between outputs

        Returns:
            StabilityMetrics object
        """
        # Run baseline
        baseline_output = execute_func(input_data)

        # Run with perturbations
        perturbed_outputs = []
        perturbed_metrics = []

        for i in range(self.num_perturbations):
            # Create perturbation
            perturbed_input = self._perturb_input(input_data, seed=i)

            # Execute
            output = execute_func(perturbed_input)
            perturbed_outputs.append(output)

            # Compute divergence from baseline
            divergence = metric_func(output, baseline_output)
            perturbed_metrics.append(divergence)

        # Compute stability metrics
        sensitivity = np.mean(perturbed_metrics)
        variance = np.var(perturbed_metrics)

        # Approximate Lyapunov exponent
        # Positive = chaos, negative = stability
        if sensitivity > 0:
            lyapunov = np.log(sensitivity / self.epsilon) if sensitivity > self.epsilon else -1.0
        else:
            lyapunov = -1.0

        # Trajectory entropy (if outputs contain trajectory info)
        trajectory_entropy = self._compute_trajectory_entropy(perturbed_outputs)

        # Determine if stable
        is_stable = (sensitivity < 0.1 and variance < 0.05 and lyapunov < 0)

        return StabilityMetrics(
            sensitivity=sensitivity,
            variance=variance,
            trajectory_entropy=trajectory_entropy,
            lyapunov_exponent=lyapunov,
            is_stable=is_stable
        )

    def _perturb_input(self, input_data: Any, seed: int = 0) -> Any:
        """
        Apply small perturbation to input.

        Args:
            input_data: Original input
            seed: Random seed for perturbation

        Returns:
            Perturbed input
        """
        np.random.seed(seed)

        if isinstance(input_data, dict) and 'train' in input_data:
            # ARC task data
            perturbed = input_data.copy()

            # Perturb one training example slightly
            if perturbed['train']:
                train_grids = perturbed['train']
                idx = np.random.randint(0, len(train_grids))
                input_grid, output_grid = train_grids[idx]

                # Add small noise to a few cells
                perturbed_grid = input_grid.copy()
                num_cells_to_perturb = max(1, int(input_grid.size * self.epsilon))

                for _ in range(num_cells_to_perturb):
                    i = np.random.randint(0, input_grid.shape[0])
                    j = np.random.randint(0, input_grid.shape[1])
                    perturbed_grid[i, j] = (perturbed_grid[i, j] + np.random.randint(1, 10)) % 10

                train_grids[idx] = (perturbed_grid, output_grid)

            return perturbed

        # Default: return unchanged (deterministic nodes won't be affected)
        return input_data

    def _compute_trajectory_entropy(self, outputs: List[Any]) -> float:
        """
        Compute entropy of trajectory patterns.

        Args:
            outputs: List of outputs from perturbed runs

        Returns:
            Entropy value
        """
        if not outputs:
            return 0.0

        # Try to extract trajectory information
        node_sequences = []

        for output in outputs:
            if hasattr(output, 'nodes'):
                # It's a Trajectory object
                node_sequences.append(tuple(output.nodes))
            elif isinstance(output, dict) and 'nodes' in output:
                node_sequences.append(tuple(output['nodes']))

        if not node_sequences:
            # No trajectory info, use output diversity as proxy
            # Count unique outputs
            unique_count = len(set(str(o) for o in outputs))
            return unique_count / len(outputs)

        # Count unique sequences
        unique_sequences = len(set(node_sequences))

        # Entropy: high if many different sequences, low if consistent
        if len(node_sequences) == 0:
            return 0.0

        probability = unique_sequences / len(node_sequences)
        entropy = -probability * np.log(probability + 1e-10) if probability > 0 else 0.0

        return entropy

    def sensitivity_sweep(
        self,
        execute_func: Callable,
        input_data: Any,
        parameter_name: str,
        parameter_values: List[float],
        metric_func: Callable
    ) -> Dict[str, Any]:
        """
        Perform sensitivity sweep over a parameter.

        Args:
            execute_func: Function to execute
            input_data: Base input
            parameter_name: Name of parameter to vary
            parameter_values: List of values to try
            metric_func: Metric function

        Returns:
            Dictionary with sweep results
        """
        results = []

        for value in parameter_values:
            # Execute with parameter value
            # Note: This is a simplified version - real implementation
            # would need to inject the parameter properly
            output = execute_func(input_data)

            # Compute metric
            # (We'd need baseline to compare against)
            results.append({
                'parameter_value': value,
                'output': output,
            })

        # Compute sensitivity as variance of results
        # This is simplified - real version would be more sophisticated
        sensitivity = len(set(str(r['output']) for r in results)) / len(results)

        return {
            'parameter': parameter_name,
            'values': parameter_values,
            'results': results,
            'sensitivity': sensitivity,
        }
