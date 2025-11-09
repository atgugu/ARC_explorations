"""
Targeted Program Synthesizer: Generates specific programs based on inferred rules.
Unlike V2's random generation, this creates THE program that implements the rule.
"""

import numpy as np
from typing import Dict, Any, List, Callable
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.node import Node, NodeOutput
from core.dsl import DSLRegistry, Operation


class TargetedSynthesizer:
    """
    Synthesizes programs targeted at implementing a specific inferred rule.
    Generates 1-5 programs (not 100!), including the right one.
    """

    def __init__(self):
        self.dsl = DSLRegistry()

    def synthesize(self, rule: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Synthesize programs that implement the given rule.

        Args:
            rule: Inferred rule from rule_inferencer

        Returns:
            List of 1-5 targeted programs
        """
        programs = []

        # Always include identity as baseline (V1 lesson!)
        programs.append({
            'type': 'identity',
            'function': self.dsl.get('identity'),
            'description': 'Identity (baseline)',
            'confidence': 0.5,
            'operations': ['identity'],
        })

        # Generate rule-specific programs
        rule_type = rule.get('rule_type', 'unknown')
        confidence = rule.get('confidence', 0.0)
        parameters = rule.get('parameters', {})

        if rule_type == 'identity':
            # Only identity program needed
            programs[0]['confidence'] = confidence
            return programs

        elif rule_type == 'rotate':
            angle = parameters.get('angle', 90)
            if angle == 90:
                op = self.dsl.get('rotate_90')
            elif angle == 180:
                op = self.dsl.get('rotate_180')
            elif angle == 270:
                op = self.dsl.get('rotate_270')
            else:
                op = None

            if op:
                programs.insert(0, {  # Insert at front (higher priority than identity)
                    'type': f'rotate_{angle}',
                    'function': op,
                    'description': f'Rotate {angle}° clockwise (learned from rule)',
                    'confidence': confidence,
                    'operations': [f'rotate_{angle}'],
                })

        elif rule_type == 'flip':
            axis = parameters.get('axis', 'horizontal')
            if axis == 'horizontal':
                op = self.dsl.get('flip_h')
                op_name = 'flip_h'
            else:  # vertical
                op = self.dsl.get('flip_v')
                op_name = 'flip_v'

            programs.insert(0, {
                'type': op_name,
                'function': op,
                'description': f'Flip {axis}ly (learned from rule)',
                'confidence': confidence,
                'operations': [op_name],
            })

        elif rule_type == 'transpose':
            op = self.dsl.get('transpose')
            programs.insert(0, {
                'type': 'transpose',
                'function': op,
                'description': 'Transpose (learned from rule)',
                'confidence': confidence,
                'operations': ['transpose'],
            })

        elif rule_type == 'color_remap':
            mapping = parameters.get('mapping', {})

            # Create custom color remap function
            def color_remap(grid: np.ndarray) -> np.ndarray:
                result = grid.copy()
                for from_color, to_color in mapping.items():
                    result[grid == int(from_color)] = int(to_color)
                return result

            programs.insert(0, {
                'type': 'learned_color_remap',
                'function': color_remap,
                'description': f'Color remap (learned): {mapping}',
                'confidence': confidence,
                'operations': ['color_remap'],
            })

        elif rule_type == 'translate':
            offset = parameters.get('offset', (0, 0))
            dy, dx = offset

            # Create custom translation function
            def translate(grid: np.ndarray) -> np.ndarray:
                result = np.zeros_like(grid)

                # Source region
                src_y_start = max(0, -dy)
                src_y_end = min(grid.shape[0], grid.shape[0] - dy)
                src_x_start = max(0, -dx)
                src_x_end = min(grid.shape[1], grid.shape[1] - dx)

                # Destination region
                dst_y_start = max(0, dy)
                dst_y_end = dst_y_start + (src_y_end - src_y_start)
                dst_x_start = max(0, dx)
                dst_x_end = dst_x_start + (src_x_end - src_x_start)

                if src_y_end > src_y_start and src_x_end > src_x_start:
                    result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                        grid[src_y_start:src_y_end, src_x_start:src_x_end]

                return result

            programs.insert(0, {
                'type': 'learned_translate',
                'function': translate,
                'description': f'Translate by ({dy}, {dx}) (learned from rule)',
                'confidence': confidence,
                'operations': ['translate'],
            })

        elif rule_type == 'scale':
            factor = parameters.get('factor', 2)
            if factor == 2:
                op = self.dsl.get('scale_2x')
            elif factor == 3:
                op = self.dsl.get('scale_3x')
            else:
                op = None

            if op:
                programs.insert(0, {
                    'type': f'scale_{factor}x',
                    'function': op,
                    'description': f'Scale {factor}x (learned from rule)',
                    'confidence': confidence,
                    'operations': [f'scale_{factor}x'],
                })

        elif rule_type == 'downsample':
            factor = parameters.get('factor', 2)
            if factor == 2:
                op = self.dsl.get('downsample_2x')
            elif factor == 3:
                op = self.dsl.get('downsample_3x')
            else:
                op = None

            if op:
                programs.insert(0, {
                    'type': f'downsample_{factor}x',
                    'function': op,
                    'description': f'Downsample {factor}x (learned from rule)',
                    'confidence': confidence,
                    'operations': [f'downsample_{factor}x'],
                })

        elif rule_type == 'tile':
            tile_h = parameters.get('tile_h', 2)
            tile_w = parameters.get('tile_w', 2)

            # Create custom tile function
            def tile_func(grid: np.ndarray) -> np.ndarray:
                return np.tile(grid, (tile_h, tile_w))

            programs.insert(0, {
                'type': f'tile_{tile_h}x{tile_w}',
                'function': tile_func,
                'description': f'Tile {tile_h}x{tile_w} (learned from rule)',
                'confidence': confidence,
                'operations': ['tile'],
            })

        elif rule_type == 'composite_or_unknown':
            # Low consistency - rely on identity + repairs
            # Keep only identity
            pass

        else:
            # Unknown rule - keep only identity
            pass

        # Sort by confidence (highest first)
        programs.sort(key=lambda p: p['confidence'], reverse=True)

        # Limit to top 5 (usually will be 1-3)
        return programs[:5]


def targeted_synthesizer_func(data: Dict[str, Any]) -> NodeOutput:
    """
    Targeted synthesis function for node.

    Args:
        data: Dictionary with 'rule' from rule_inferencer

    Returns:
        NodeOutput with synthesized programs
    """
    artifacts = {}
    telemetry = {'node_type': 'reasoner', 'subtype': 'targeted_synthesizer'}

    rule = data.get('rule', None)

    if rule is None:
        telemetry['success'] = False
        return NodeOutput(result=[], artifacts=artifacts, telemetry=telemetry)

    synthesizer = TargetedSynthesizer()
    programs = synthesizer.synthesize(rule)

    artifacts['programs'] = programs
    telemetry['num_programs'] = len(programs)
    telemetry['rule_type'] = rule.get('rule_type', 'unknown')
    telemetry['success'] = True

    return NodeOutput(result=programs, artifacts=artifacts, telemetry=telemetry)


def create_targeted_synthesizer_node() -> Node:
    """Create a targeted synthesizer node."""
    return Node(
        name="targeted_synthesizer",
        func=targeted_synthesizer_func,
        input_type="rule",
        output_type="programs",
        deterministic=True,
        category="reasoner"
    )
