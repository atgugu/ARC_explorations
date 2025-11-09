"""
Behavior vector computation and analysis.
"""

import numpy as np
from typing import Dict, List, Any
import hashlib


class BehaviorVectorBuilder:
    """
    Builds behavior vectors for nodes based on probe bank execution.
    """

    def __init__(self, probe_bank: List[Dict[str, Any]]):
        """
        Initialize with a probe bank.

        Args:
            probe_bank: List of small test cases to profile nodes
        """
        self.probe_bank = probe_bank
        self.vector_cache: Dict[str, np.ndarray] = {}

    def compute_behavior_vector(self, node, node_registry, vector_dim: int = 256) -> np.ndarray:
        """
        Compute behavior vector for a node by running it on probe bank.

        Args:
            node: Node to profile
            node_registry: NodeRegistry for execution
            vector_dim: Target dimensionality

        Returns:
            Behavior vector (numpy array)
        """
        # Check cache
        cache_key = node.name
        if cache_key in self.vector_cache:
            return self.vector_cache[cache_key]

        features = []

        # Run node on each probe
        for probe in self.probe_bank:
            try:
                output = node_registry.execute(node.name, probe, use_cache=False)

                # Extract features from output
                probe_features = self._extract_output_features(output)
                features.extend(probe_features)

            except Exception:
                # If node fails on probe, add zeros
                features.extend([0.0] * 10)

        # Pad or truncate to target dimension
        features = np.array(features, dtype=np.float32)

        if len(features) < vector_dim:
            # Pad with zeros
            padding = np.zeros(vector_dim - len(features), dtype=np.float32)
            features = np.concatenate([features, padding])
        else:
            # Truncate
            features = features[:vector_dim]

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        # Cache and return
        self.vector_cache[cache_key] = features
        return features

    def _extract_output_features(self, output) -> List[float]:
        """
        Extract numerical features from node output.

        Args:
            output: NodeOutput object

        Returns:
            List of feature values
        """
        features = []

        # Success flag
        features.append(1.0 if output.telemetry.get('success', False) else 0.0)

        # Result type indicator
        if output.result is None:
            features.append(0.0)
        elif isinstance(output.result, (int, float)):
            features.append(float(output.result))
        elif isinstance(output.result, (list, dict)):
            features.append(len(output.result))
        else:
            features.append(1.0)

        # Artifact count
        features.append(len(output.artifacts))

        # Hash of output for uniqueness
        result_hash = self._hash_output(output.result)
        # Convert hash to a few numerical features
        hash_int = int(result_hash[:8], 16)
        features.append((hash_int % 256) / 255.0)
        features.append(((hash_int >> 8) % 256) / 255.0)

        # Extract some artifact statistics if available
        if 'num_hypotheses' in output.telemetry:
            features.append(output.telemetry['num_hypotheses'] / 10.0)
        else:
            features.append(0.0)

        if 'num_programs' in output.telemetry:
            features.append(output.telemetry['num_programs'] / 10.0)
        else:
            features.append(0.0)

        # Category encoding
        node_type = output.telemetry.get('node_type', 'generic')
        type_encoding = {
            'extractor': 1.0,
            'reasoner': 2.0,
            'executor': 3.0,
            'critic': 4.0,
            'generic': 0.0,
        }
        features.append(type_encoding.get(node_type, 0.0) / 4.0)

        return features

    def _hash_output(self, result: Any) -> str:
        """Compute hash of output result."""
        try:
            if isinstance(result, np.ndarray):
                data = result.tobytes()
            elif isinstance(result, (list, dict)):
                import json
                data = json.dumps(result, sort_keys=True).encode()
            else:
                data = str(result).encode()

            return hashlib.md5(data).hexdigest()
        except:
            return "00000000"

    def compute_all_vectors(self, node_registry, vector_dim: int = 256) -> Dict[str, np.ndarray]:
        """
        Compute behavior vectors for all nodes in registry.

        Args:
            node_registry: NodeRegistry
            vector_dim: Target dimensionality

        Returns:
            Dictionary mapping node names to behavior vectors
        """
        vectors = {}

        for node_name, node in node_registry.nodes.items():
            vectors[node_name] = self.compute_behavior_vector(node, node_registry, vector_dim)

        return vectors


def create_simple_probe_bank() -> List[Dict[str, Any]]:
    """
    Create a simple probe bank for behavior vector computation.

    Returns:
        List of probe tasks
    """
    probes = []

    # Simple identity probes
    for size in [(3, 3), (5, 5), (4, 6)]:
        grid = np.random.randint(0, 10, size=size)
        probes.append({
            'train': [(grid, grid.copy())],
            'test': [(grid, grid.copy())]
        })

    # Symmetry probes
    for size in [(4, 4), (6, 6)]:
        grid = np.random.randint(0, 10, size=size)
        symmetric = (grid + grid.T) // 2  # Make symmetric
        probes.append({
            'train': [(grid, symmetric)],
            'test': [(grid, symmetric)]
        })

    # Color remap probes
    for size in [(3, 3), (5, 5)]:
        grid = np.random.randint(0, 5, size=size)
        remapped = (grid + 1) % 10
        probes.append({
            'train': [(grid, remapped)],
            'test': [(grid, remapped)]
        })

    # Shape change probes
    grid = np.random.randint(0, 10, size=(3, 3))
    larger = np.tile(grid, (2, 2))
    probes.append({
        'train': [(grid, larger)],
        'test': [(grid, larger)]
    })

    # Rotation probes
    for size in [(4, 4), (6, 6)]:
        grid = np.random.randint(0, 10, size=size)
        rotated = np.rot90(grid)
        probes.append({
            'train': [(grid, rotated)],
            'test': [(grid, rotated)]
        })

    return probes
