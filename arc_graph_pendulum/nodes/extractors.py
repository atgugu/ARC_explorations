"""
Feature extractor nodes for perception phase.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import ndimage
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.node import Node, NodeOutput
from utils.grid_utils import Grid, extract_bounding_box, detect_periodicity


def color_histogram_func(task_data: Dict[str, Any]) -> NodeOutput:
    """
    Extract color histograms from all grids in a task.

    Args:
        task_data: Dictionary with 'train' and 'test' grids

    Returns:
        NodeOutput with color statistics
    """
    artifacts = {}
    telemetry = {'node_type': 'extractor', 'subtype': 'color_histogram'}

    train_grids = task_data.get('train', [])

    # Compute histograms for each training example
    train_histograms = []
    for input_grid, output_grid in train_grids:
        input_hist = Grid(input_grid).get_color_histogram()
        output_hist = Grid(output_grid).get_color_histogram()

        train_histograms.append({
            'input_hist': input_hist,
            'output_hist': output_hist,
            'color_changes': output_hist - input_hist,
        })

    # Identify consistent color transformations
    if train_histograms:
        avg_changes = np.mean([h['color_changes'] for h in train_histograms], axis=0)
        artifacts['avg_color_changes'] = avg_changes
        artifacts['train_histograms'] = train_histograms

        # Identify dominant colors
        all_colors = set()
        for h in train_histograms:
            all_colors.update(np.where(h['input_hist'] > 0)[0])
            all_colors.update(np.where(h['output_hist'] > 0)[0])

        artifacts['used_colors'] = sorted(all_colors)

    telemetry['success'] = True
    return NodeOutput(result=artifacts, artifacts=artifacts, telemetry=telemetry)


def object_detector_func(task_data: Dict[str, Any]) -> NodeOutput:
    """
    Detect and extract objects (connected components) from grids.

    Args:
        task_data: Dictionary with 'train' and 'test' grids

    Returns:
        NodeOutput with object information
    """
    artifacts = {}
    telemetry = {'node_type': 'extractor', 'subtype': 'object_detector'}

    train_grids = task_data.get('train', [])

    train_objects = []
    for input_grid, output_grid in train_grids:
        # Find objects in input
        input_objects = []
        for color in range(1, 10):  # Skip background (0)
            mask = input_grid == color
            if mask.any():
                labeled, num_features = ndimage.label(mask)

                for i in range(1, num_features + 1):
                    obj_mask = labeled == i
                    bbox = extract_bounding_box(input_grid, color)

                    input_objects.append({
                        'color': color,
                        'mask': obj_mask,
                        'bbox': bbox,
                        'size': int(obj_mask.sum()),
                        'grid_section': input_grid[obj_mask].copy()
                    })

        # Find objects in output
        output_objects = []
        for color in range(1, 10):
            mask = output_grid == color
            if mask.any():
                labeled, num_features = ndimage.label(mask)

                for i in range(1, num_features + 1):
                    obj_mask = labeled == i
                    bbox = extract_bounding_box(output_grid, color)

                    output_objects.append({
                        'color': color,
                        'mask': obj_mask,
                        'bbox': bbox,
                        'size': int(obj_mask.sum()),
                        'grid_section': output_grid[obj_mask].copy()
                    })

        train_objects.append({
            'input_objects': input_objects,
            'output_objects': output_objects,
            'num_input': len(input_objects),
            'num_output': len(output_objects),
        })

    artifacts['train_objects'] = train_objects
    artifacts['avg_input_objects'] = np.mean([o['num_input'] for o in train_objects]) if train_objects else 0
    artifacts['avg_output_objects'] = np.mean([o['num_output'] for o in train_objects]) if train_objects else 0

    telemetry['success'] = True
    return NodeOutput(result=artifacts, artifacts=artifacts, telemetry=telemetry)


def symmetry_detector_func(task_data: Dict[str, Any]) -> NodeOutput:
    """
    Detect symmetries in grids.

    Args:
        task_data: Dictionary with 'train' and 'test' grids

    Returns:
        NodeOutput with symmetry information
    """
    artifacts = {}
    telemetry = {'node_type': 'extractor', 'subtype': 'symmetry_detector'}

    train_grids = task_data.get('train', [])

    symmetries = []
    for input_grid, output_grid in train_grids:
        input_g = Grid(input_grid)
        output_g = Grid(output_grid)

        sym = {
            'input_vertical': input_g.has_symmetry_vertical(),
            'input_horizontal': input_g.has_symmetry_horizontal(),
            'input_diagonal': input_g.has_symmetry_diagonal(),
            'output_vertical': output_g.has_symmetry_vertical(),
            'output_horizontal': output_g.has_symmetry_horizontal(),
            'output_diagonal': output_g.has_symmetry_diagonal(),
        }

        symmetries.append(sym)

    artifacts['symmetries'] = symmetries

    # Check if symmetries are consistent
    if symmetries:
        # Count how many examples have each symmetry
        for key in ['input_vertical', 'input_horizontal', 'output_vertical', 'output_horizontal']:
            count = sum(s[key] for s in symmetries)
            artifacts[f'{key}_count'] = count
            artifacts[f'{key}_consistent'] = (count == 0 or count == len(symmetries))

    telemetry['success'] = True
    return NodeOutput(result=artifacts, artifacts=artifacts, telemetry=telemetry)


def periodicity_detector_func(task_data: Dict[str, Any]) -> NodeOutput:
    """
    Detect periodic patterns in grids.

    Args:
        task_data: Dictionary with 'train' and 'test' grids

    Returns:
        NodeOutput with periodicity information
    """
    artifacts = {}
    telemetry = {'node_type': 'extractor', 'subtype': 'periodicity_detector'}

    train_grids = task_data.get('train', [])

    periodicities = []
    for input_grid, output_grid in train_grids:
        input_period = detect_periodicity(input_grid)
        output_period = detect_periodicity(output_grid)

        periodicities.append({
            'input_period': input_period,
            'output_period': output_period,
            'is_periodic': any(input_period) or any(output_period),
        })

    artifacts['periodicities'] = periodicities

    # Check if periodicity is consistent
    if periodicities:
        periodic_count = sum(p['is_periodic'] for p in periodicities)
        artifacts['periodic_count'] = periodic_count
        artifacts['is_periodic_task'] = periodic_count >= len(periodicities) / 2

    telemetry['success'] = True
    return NodeOutput(result=artifacts, artifacts=artifacts, telemetry=telemetry)


def shape_detector_func(task_data: Dict[str, Any]) -> NodeOutput:
    """
    Detect basic shapes and patterns.

    Args:
        task_data: Dictionary with 'train' and 'test' grids

    Returns:
        NodeOutput with shape information
    """
    artifacts = {}
    telemetry = {'node_type': 'extractor', 'subtype': 'shape_detector'}

    train_grids = task_data.get('train', [])

    shapes = []
    for input_grid, output_grid in train_grids:
        shape_info = {
            'input_shape': input_grid.shape,
            'output_shape': output_grid.shape,
            'shape_preserved': input_grid.shape == output_grid.shape,
            'size_ratio': (output_grid.shape[0] * output_grid.shape[1]) / (input_grid.shape[0] * input_grid.shape[1]) if input_grid.size > 0 else 0,
        }

        shapes.append(shape_info)

    artifacts['shapes'] = shapes

    # Check if shape transformation is consistent
    if shapes:
        shape_preserved_count = sum(s['shape_preserved'] for s in shapes)
        artifacts['shape_preserved_count'] = shape_preserved_count
        artifacts['always_preserves_shape'] = (shape_preserved_count == len(shapes))

        if not artifacts['always_preserves_shape']:
            # Compute average size ratio
            avg_ratio = np.mean([s['size_ratio'] for s in shapes])
            artifacts['avg_size_ratio'] = avg_ratio

    telemetry['success'] = True
    return NodeOutput(result=artifacts, artifacts=artifacts, telemetry=telemetry)


# Node factory functions
def create_color_histogram_node() -> Node:
    """Create a color histogram extractor node."""
    return Node(
        name="color_histogram",
        func=color_histogram_func,
        input_type="task_data",
        output_type="color_stats",
        deterministic=True,
        category="extractor"
    )


def create_object_detector_node() -> Node:
    """Create an object detector node."""
    return Node(
        name="object_detector",
        func=object_detector_func,
        input_type="task_data",
        output_type="objects",
        deterministic=True,
        category="extractor"
    )


def create_symmetry_detector_node() -> Node:
    """Create a symmetry detector node."""
    return Node(
        name="symmetry_detector",
        func=symmetry_detector_func,
        input_type="task_data",
        output_type="symmetries",
        deterministic=True,
        category="extractor"
    )


def create_periodicity_detector_node() -> Node:
    """Create a periodicity detector node."""
    return Node(
        name="periodicity_detector",
        func=periodicity_detector_func,
        input_type="task_data",
        output_type="periodicity",
        deterministic=True,
        category="extractor"
    )


def create_shape_detector_node() -> Node:
    """Create a shape detector node."""
    return Node(
        name="shape_detector",
        func=shape_detector_func,
        input_type="task_data",
        output_type="shapes",
        deterministic=True,
        category="extractor"
    )
