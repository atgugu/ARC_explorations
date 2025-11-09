"""
Node abstraction: typed skill/specialist functions with caching and contracts.
"""

from typing import Any, Dict, List, Callable, Tuple, Optional
from dataclasses import dataclass, field
import hashlib
import json
import numpy as np


@dataclass
class NodeOutput:
    """Output from a node execution."""
    result: Any
    artifacts: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    """
    A typed function representing a skill/specialist.

    Attributes:
        name: Unique identifier for the node
        func: The actual function to execute
        input_type: Expected input type description
        output_type: Expected output type description
        deterministic: Whether the node is deterministic (for caching)
        category: Node category (extractor, reasoner, executor, critic)
    """
    name: str
    func: Callable
    input_type: str
    output_type: str
    deterministic: bool = True
    category: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __call__(self, input_data: Any, **kwargs) -> NodeOutput:
        """Execute the node function."""
        try:
            result = self.func(input_data, **kwargs)

            # Ensure result is wrapped in NodeOutput
            if isinstance(result, NodeOutput):
                return result
            elif isinstance(result, tuple) and len(result) == 3:
                return NodeOutput(result=result[0], artifacts=result[1], telemetry=result[2])
            else:
                return NodeOutput(result=result)

        except Exception as e:
            return NodeOutput(
                result=None,
                artifacts={},
                telemetry={'error': str(e), 'success': False}
            )

    def compute_hash(self, input_data: Any) -> str:
        """Compute hash of input for caching."""
        try:
            # Handle numpy arrays
            if isinstance(input_data, np.ndarray):
                data_str = input_data.tobytes().hex()
            # Handle lists/dicts
            elif isinstance(input_data, (list, dict)):
                data_str = json.dumps(input_data, sort_keys=True)
            else:
                data_str = str(input_data)

            combined = f"{self.name}:{data_str}"
            return hashlib.sha256(combined.encode()).hexdigest()
        except:
            # Fallback for unhashable types
            return hashlib.sha256(f"{self.name}:{id(input_data)}".encode()).hexdigest()


class NodeRegistry:
    """
    Registry for managing nodes with caching.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.cache: Dict[str, NodeOutput] = {}
        self.execution_counts: Dict[str, int] = {}
        self.behavior_vectors: Dict[str, np.ndarray] = {}

    def register(self, node: Node):
        """Register a node in the registry."""
        self.nodes[node.name] = node
        self.execution_counts[node.name] = 0

    def get(self, name: str) -> Optional[Node]:
        """Get a node by name."""
        return self.nodes.get(name)

    def execute(self, node_name: str, input_data: Any, use_cache: bool = True, **kwargs) -> NodeOutput:
        """
        Execute a node with optional caching.

        Args:
            node_name: Name of the node to execute
            input_data: Input data for the node
            use_cache: Whether to use cached results
            **kwargs: Additional arguments to pass to the node

        Returns:
            NodeOutput from the execution
        """
        node = self.get(node_name)
        if not node:
            return NodeOutput(
                result=None,
                telemetry={'error': f'Node {node_name} not found', 'success': False}
            )

        # Check cache if deterministic and caching enabled
        cache_key = None
        if use_cache and node.deterministic:
            cache_key = node.compute_hash(input_data)
            if cache_key in self.cache:
                cached_output = self.cache[cache_key]
                cached_output.telemetry['cache_hit'] = True
                return cached_output

        # Execute the node
        output = node(input_data, **kwargs)
        self.execution_counts[node_name] += 1
        output.telemetry['node_name'] = node_name
        output.telemetry['execution_count'] = self.execution_counts[node_name]

        # Cache result if applicable
        if use_cache and node.deterministic and cache_key:
            self.cache[cache_key] = output

        return output

    def get_nodes_by_category(self, category: str) -> List[Node]:
        """Get all nodes in a specific category."""
        return [node for node in self.nodes.values() if node.category == category]

    def clear_cache(self):
        """Clear the execution cache."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            'total_nodes': len(self.nodes),
            'cache_size': len(self.cache),
            'execution_counts': self.execution_counts.copy(),
            'categories': {
                cat: len(self.get_nodes_by_category(cat))
                for cat in set(node.category for node in self.nodes.values())
            }
        }
