"""
Rule Inferencer: Infers general transformation rules from differential analyses.
Abstracts specific observations into a unified rule that can guide synthesis.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import Counter
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.node import Node, NodeOutput


class RuleInferencer:
    """
    Infers transformation rules from multiple example analyses.
    """

    def infer_rule(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Infer a general rule from differential analyses.

        Args:
            analyses: List of transformation analyses from differential_analyzer

        Returns:
            Dictionary with:
            - rule_type: Main transformation category
            - parameters: Learned parameters
            - consistency: How consistent across examples (0-1)
            - confidence: Overall confidence in rule
            - description: Human-readable rule description
        """
        if not analyses:
            return self._unknown_rule()

        # Find consensus on transformation type
        types = [a['transformation_type'] for a in analyses]
        type_counter = Counter(types)
        most_common_type, count = type_counter.most_common(1)[0]

        consistency = count / len(types)

        # If low consistency, might be composite rule
        if consistency < 0.6:
            return self._infer_composite_rule(analyses)

        # High consistency - simple rule
        return self._infer_simple_rule(most_common_type, analyses, consistency)

    def _infer_simple_rule(
        self,
        rule_type: str,
        analyses: List[Dict],
        consistency: float
    ) -> Dict[str, Any]:
        """Infer a simple (single-operation) rule."""

        # Filter to only analyses of this type
        relevant = [a for a in analyses if a['transformation_type'] == rule_type]

        if rule_type == 'identity':
            return {
                'rule_type': 'identity',
                'parameters': {},
                'consistency': consistency,
                'confidence': consistency,
                'description': 'Output is identical to input'
            }

        elif rule_type == 'rotate':
            # Check if all rotations are by same angle
            angles = [a['parameters']['angle'] for a in relevant]
            if len(set(angles)) == 1:
                angle = angles[0]
                return {
                    'rule_type': 'rotate',
                    'parameters': {'angle': angle},
                    'consistency': consistency,
                    'confidence': consistency,
                    'description': f'Rotate {angle}° clockwise'
                }
            else:
                # Different angles - inconsistent
                return {
                    'rule_type': 'rotate',
                    'parameters': {'angle': Counter(angles).most_common(1)[0][0]},
                    'consistency': consistency * 0.5,
                    'confidence': consistency * 0.5,
                    'description': 'Rotation with inconsistent angles'
                }

        elif rule_type == 'flip':
            # Check if all flips are on same axis
            axes = [a['parameters']['axis'] for a in relevant]
            if len(set(axes)) == 1:
                axis = axes[0]
                return {
                    'rule_type': 'flip',
                    'parameters': {'axis': axis},
                    'consistency': consistency,
                    'confidence': consistency,
                    'description': f'Flip {axis}ly'
                }
            else:
                return {
                    'rule_type': 'flip',
                    'parameters': {'axis': Counter(axes).most_common(1)[0][0]},
                    'consistency': consistency * 0.5,
                    'confidence': consistency * 0.5,
                    'description': 'Flip with inconsistent axes'
                }

        elif rule_type == 'transpose':
            return {
                'rule_type': 'transpose',
                'parameters': {},
                'consistency': consistency,
                'confidence': consistency,
                'description': 'Transpose grid'
            }

        elif rule_type == 'color_remap':
            # Merge color mappings from all examples
            merged_mapping = self._merge_color_mappings(relevant)
            return {
                'rule_type': 'color_remap',
                'parameters': {'mapping': merged_mapping},
                'consistency': consistency,
                'confidence': consistency * 0.9,  # Slightly lower for learned mappings
                'description': f'Remap colors: {merged_mapping}'
            }

        elif rule_type == 'translate':
            # Check if all translations use same offset
            offsets = [tuple(a['parameters']['offset']) for a in relevant]
            if len(set(offsets)) == 1:
                offset = offsets[0]
                return {
                    'rule_type': 'translate',
                    'parameters': {'offset': offset},
                    'consistency': consistency,
                    'confidence': consistency,
                    'description': f'Translate by {offset}'
                }
            else:
                # Different offsets - use most common
                most_common_offset = Counter(offsets).most_common(1)[0][0]
                return {
                    'rule_type': 'translate',
                    'parameters': {'offset': most_common_offset},
                    'consistency': consistency * 0.7,
                    'confidence': consistency * 0.7,
                    'description': f'Translate (inconsistent offsets, using {most_common_offset})'
                }

        elif rule_type == 'scale':
            # Check scale factors
            factors = [a['parameters']['factor'] for a in relevant]
            if len(set(factors)) == 1:
                factor = factors[0]
                return {
                    'rule_type': 'scale',
                    'parameters': {'factor': factor},
                    'consistency': consistency,
                    'confidence': consistency,
                    'description': f'Scale {factor}x'
                }
            else:
                return {
                    'rule_type': 'scale',
                    'parameters': {'factor': Counter(factors).most_common(1)[0][0]},
                    'consistency': consistency * 0.6,
                    'confidence': consistency * 0.6,
                    'description': 'Scale with inconsistent factors'
                }

        elif rule_type == 'downsample':
            factors = [a['parameters']['factor'] for a in relevant]
            if len(set(factors)) == 1:
                factor = factors[0]
                return {
                    'rule_type': 'downsample',
                    'parameters': {'factor': factor},
                    'consistency': consistency,
                    'confidence': consistency,
                    'description': f'Downsample {factor}x'
                }
            else:
                return {
                    'rule_type': 'downsample',
                    'parameters': {'factor': Counter(factors).most_common(1)[0][0]},
                    'consistency': consistency * 0.6,
                    'confidence': consistency * 0.6,
                    'description': 'Downsample with inconsistent factors'
                }

        elif rule_type == 'tile':
            # Check tiling parameters
            tile_params = [(a['parameters']['tile_h'], a['parameters']['tile_w']) for a in relevant]
            if len(set(tile_params)) == 1:
                tile_h, tile_w = tile_params[0]
                return {
                    'rule_type': 'tile',
                    'parameters': {'tile_h': tile_h, 'tile_w': tile_w},
                    'consistency': consistency,
                    'confidence': consistency,
                    'description': f'Tile {tile_h}x{tile_w}'
                }
            else:
                most_common = Counter(tile_params).most_common(1)[0][0]
                return {
                    'rule_type': 'tile',
                    'parameters': {'tile_h': most_common[0], 'tile_w': most_common[1]},
                    'consistency': consistency * 0.7,
                    'confidence': consistency * 0.7,
                    'description': f'Tile with inconsistent parameters'
                }

        else:
            return self._unknown_rule()

    def _merge_color_mappings(self, analyses: List[Dict]) -> Dict[int, int]:
        """
        Merge color mappings from multiple examples.
        Uses majority voting for conflicts.
        """
        # Collect all mappings
        vote_counts = {}  # (from_color, to_color) -> count

        for analysis in analyses:
            mapping = analysis['parameters']['mapping']
            for from_color, to_color in mapping.items():
                key = (int(from_color), int(to_color))
                vote_counts[key] = vote_counts.get(key, 0) + 1

        # For each from_color, find most common to_color
        merged = {}
        from_colors = set(fc for fc, _ in vote_counts.keys())

        for from_color in from_colors:
            # Get all votes for this from_color
            votes = [(tc, count) for (fc, tc), count in vote_counts.items() if fc == from_color]
            # Pick most common
            if votes:
                best_to_color = max(votes, key=lambda x: x[1])[0]
                merged[from_color] = best_to_color

        return merged

    def _infer_composite_rule(self, analyses: List[Dict]) -> Dict[str, Any]:
        """
        Infer a composite rule when consistency is low.
        For now, fallback to identity + repairs approach.
        """
        # Count transformation types
        type_counter = Counter(a['transformation_type'] for a in analyses)

        # If multiple types, it might be a composite transformation
        # For now, return low-confidence result suggesting identity + repairs

        return {
            'rule_type': 'composite_or_unknown',
            'parameters': {
                'observed_types': dict(type_counter),
                'suggestion': 'Try identity + repairs'
            },
            'consistency': 0.3,
            'confidence': 0.2,
            'description': f'Low consistency across examples. Observed: {dict(type_counter)}'
        }

    def _unknown_rule(self) -> Dict[str, Any]:
        """Return unknown rule with low confidence."""
        return {
            'rule_type': 'unknown',
            'parameters': {},
            'consistency': 0.0,
            'confidence': 0.0,
            'description': 'No clear transformation rule detected'
        }


def rule_inferencer_func(data: Dict[str, Any]) -> NodeOutput:
    """
    Rule inference function for node.

    Args:
        data: Dictionary with 'transformation_analyses' from differential_analyzer

    Returns:
        NodeOutput with inferred rule
    """
    artifacts = {}
    telemetry = {'node_type': 'reasoner', 'subtype': 'rule_inferencer'}

    analyses = data.get('transformation_analyses', [])

    if not analyses:
        telemetry['success'] = False
        rule = {
            'rule_type': 'unknown',
            'parameters': {},
            'consistency': 0.0,
            'confidence': 0.0,
            'description': 'No analyses provided'
        }
        return NodeOutput(result=rule, artifacts={'rule': rule}, telemetry=telemetry)

    inferencer = RuleInferencer()
    rule = inferencer.infer_rule(analyses)

    artifacts['rule'] = rule
    telemetry['rule_type'] = rule['rule_type']
    telemetry['consistency'] = rule['consistency']
    telemetry['confidence'] = rule['confidence']
    telemetry['success'] = True

    return NodeOutput(result=rule, artifacts=artifacts, telemetry=telemetry)


def create_rule_inferencer_node() -> Node:
    """Create a rule inferencer node."""
    return Node(
        name="rule_inferencer",
        func=rule_inferencer_func,
        input_type="transformation_analyses",
        output_type="rule",
        deterministic=True,
        category="reasoner"
    )
