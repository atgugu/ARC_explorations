"""
Reasoner and program synthesis nodes.
"""

import numpy as np
from typing import Dict, Any, List, Callable
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.node import Node, NodeOutput
from utils.grid_utils import Grid


def hypothesis_generator_func(facts: Dict[str, Any]) -> NodeOutput:
    """
    Generate transformation hypotheses based on extracted facts.

    Args:
        facts: Dictionary of facts from extractor nodes

    Returns:
        NodeOutput with hypothesis list
    """
    artifacts = {}
    telemetry = {'node_type': 'reasoner', 'subtype': 'hypothesis_generator'}

    hypotheses = []

    # Analyze facts and generate hypotheses
    # Shape-based hypotheses
    if 'shapes' in facts:
        shapes = facts['shapes']
        if any(s.get('shape_preserved', False) for s in shapes):
            hypotheses.append({
                'type': 'identity',
                'description': 'Output is same as input',
                'confidence': 0.5,
            })

        if any(not s.get('shape_preserved', False) for s in shapes):
            hypotheses.append({
                'type': 'resize',
                'description': 'Grid is resized',
                'confidence': 0.6,
            })

    # Color-based hypotheses
    if 'color_changes' in facts:
        hypotheses.append({
            'type': 'color_remap',
            'description': 'Colors are remapped',
            'confidence': 0.7,
        })

    # Symmetry-based hypotheses
    if 'symmetries' in facts:
        symmetries = facts['symmetries']
        if any(s.get('output_vertical', False) for s in symmetries):
            hypotheses.append({
                'type': 'mirror_vertical',
                'description': 'Apply vertical mirroring',
                'confidence': 0.8,
            })

        if any(s.get('output_horizontal', False) for s in symmetries):
            hypotheses.append({
                'type': 'mirror_horizontal',
                'description': 'Apply horizontal mirroring',
                'confidence': 0.8,
            })

    # Object-based hypotheses
    if 'train_objects' in facts:
        hypotheses.append({
            'type': 'object_transform',
            'description': 'Transform objects individually',
            'confidence': 0.6,
        })

    # Periodicity-based hypotheses
    if 'is_periodic_task' in facts and facts['is_periodic_task']:
        hypotheses.append({
            'type': 'tile_pattern',
            'description': 'Tile or repeat pattern',
            'confidence': 0.7,
        })

    # Default hypothesis
    if not hypotheses:
        hypotheses.append({
            'type': 'unknown',
            'description': 'Unknown transformation',
            'confidence': 0.1,
        })

    artifacts['hypotheses'] = hypotheses
    telemetry['num_hypotheses'] = len(hypotheses)
    telemetry['success'] = True

    return NodeOutput(result=hypotheses, artifacts=artifacts, telemetry=telemetry)


def program_synthesizer_func(data: Dict[str, Any]) -> NodeOutput:
    """
    Synthesize executable programs based on hypotheses and facts.

    Args:
        data: Dictionary with 'hypotheses' and 'facts'

    Returns:
        NodeOutput with synthesized programs
    """
    artifacts = {}
    telemetry = {'node_type': 'reasoner', 'subtype': 'program_synthesizer'}

    hypotheses = data.get('hypotheses', [])
    facts = data.get('facts', {})
    task_data = data.get('task_data', {})

    programs = []

    # Generate programs for each hypothesis
    for hyp in hypotheses:
        hyp_type = hyp.get('type', 'unknown')

        if hyp_type == 'identity':
            programs.append({
                'type': 'identity',
                'function': lambda grid: grid.copy(),
                'description': 'Return input unchanged',
                'confidence': hyp.get('confidence', 0.5),
            })

        elif hyp_type == 'mirror_vertical':
            programs.append({
                'type': 'mirror_vertical',
                'function': lambda grid: np.fliplr(grid),
                'description': 'Flip horizontally',
                'confidence': hyp.get('confidence', 0.7),
            })

        elif hyp_type == 'mirror_horizontal':
            programs.append({
                'type': 'mirror_horizontal',
                'function': lambda grid: np.flipud(grid),
                'description': 'Flip vertically',
                'confidence': hyp.get('confidence', 0.7),
            })

        elif hyp_type == 'color_remap':
            # Try to infer color mapping from training data
            if 'train_histograms' in facts:
                # Simple color swap
                programs.append({
                    'type': 'color_remap',
                    'function': lambda grid: remap_colors_simple(grid, facts),
                    'description': 'Remap colors based on training',
                    'confidence': hyp.get('confidence', 0.5),
                })

        elif hyp_type == 'tile_pattern':
            programs.append({
                'type': 'tile_pattern',
                'function': lambda grid: tile_grid(grid, 2, 2),
                'description': 'Tile pattern 2x2',
                'confidence': hyp.get('confidence', 0.6),
            })

    # Add a few baseline transformations
    programs.append({
        'type': 'rotate_90',
        'function': lambda grid: np.rot90(grid),
        'description': 'Rotate 90 degrees',
        'confidence': 0.3,
    })

    programs.append({
        'type': 'rotate_180',
        'function': lambda grid: np.rot90(grid, 2),
        'description': 'Rotate 180 degrees',
        'confidence': 0.3,
    })

    artifacts['programs'] = programs
    telemetry['num_programs'] = len(programs)
    telemetry['success'] = True

    return NodeOutput(result=programs, artifacts=artifacts, telemetry=telemetry)


# Helper functions for transformations
def remap_colors_simple(grid: np.ndarray, facts: Dict[str, Any]) -> np.ndarray:
    """Simple color remapping based on observed changes."""
    result = grid.copy()

    # Try to find consistent color mappings
    if 'avg_color_changes' in facts:
        changes = facts['avg_color_changes']

        # Find colors that consistently increase or decrease
        for i, change in enumerate(changes):
            if abs(change) > 0.5:  # Significant change
                # This is a simplification - real implementation would be smarter
                pass

    return result


def tile_grid(grid: np.ndarray, repeat_y: int, repeat_x: int) -> np.ndarray:
    """Tile a grid."""
    return np.tile(grid, (repeat_y, repeat_x))


# Node factory functions
def create_hypothesis_generator_node() -> Node:
    """Create a hypothesis generator node."""
    return Node(
        name="hypothesis_generator",
        func=hypothesis_generator_func,
        input_type="facts",
        output_type="hypotheses",
        deterministic=True,
        category="reasoner"
    )


def create_program_synthesizer_node() -> Node:
    """Create a program synthesizer node."""
    return Node(
        name="program_synthesizer",
        func=program_synthesizer_func,
        input_type="hypotheses_and_facts",
        output_type="programs",
        deterministic=True,
        category="reasoner"
    )
