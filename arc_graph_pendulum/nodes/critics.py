"""
Critic nodes for evaluating program outputs.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.node import Node, NodeOutput
from utils.grid_utils import compute_iou, compute_hamming_distance


def iou_critic_func(data: Dict[str, Any]) -> NodeOutput:
    """
    Compute IoU (Intersection over Union) score for predictions.

    Args:
        data: Dictionary with 'predictions' and 'targets'

    Returns:
        NodeOutput with IoU scores
    """
    artifacts = {}
    telemetry = {'node_type': 'critic', 'subtype': 'iou_critic'}

    predictions = data.get('predictions', [])
    targets = data.get('targets', [])

    if len(predictions) != len(targets):
        telemetry['error'] = 'Mismatch between predictions and targets'
        telemetry['success'] = False
        return NodeOutput(result=0.0, artifacts=artifacts, telemetry=telemetry)

    scores = []
    for pred, target in zip(predictions, targets):
        score = compute_iou(pred, target)
        scores.append(score)

    avg_score = np.mean(scores) if scores else 0.0

    artifacts['scores'] = scores
    artifacts['avg_score'] = avg_score
    artifacts['min_score'] = np.min(scores) if scores else 0.0
    artifacts['max_score'] = np.max(scores) if scores else 0.0

    telemetry['success'] = True
    telemetry['num_evaluated'] = len(scores)

    return NodeOutput(result=avg_score, artifacts=artifacts, telemetry=telemetry)


def failure_analyzer_func(data: Dict[str, Any]) -> NodeOutput:
    """
    Analyze failures and generate taxonomy.

    Args:
        data: Dictionary with 'predictions', 'targets', and 'inputs'

    Returns:
        NodeOutput with failure taxonomy
    """
    artifacts = {}
    telemetry = {'node_type': 'critic', 'subtype': 'failure_analyzer'}

    predictions = data.get('predictions', [])
    targets = data.get('targets', [])
    inputs = data.get('inputs', [])

    failure_types = []

    for i, (pred, target) in enumerate(zip(predictions, targets)):
        failure = analyze_single_failure(pred, target, inputs[i] if i < len(inputs) else None)
        failure_types.append(failure)

    # Aggregate failure patterns
    taxonomy = {
        'shape_mismatch': sum(1 for f in failure_types if f['type'] == 'shape_mismatch'),
        'color_error': sum(1 for f in failure_types if f['type'] == 'color_error'),
        'placement_error': sum(1 for f in failure_types if f['type'] == 'placement_error'),
        'partial_correct': sum(1 for f in failure_types if f['type'] == 'partial_correct'),
        'completely_wrong': sum(1 for f in failure_types if f['type'] == 'completely_wrong'),
        'perfect': sum(1 for f in failure_types if f['type'] == 'perfect'),
    }

    artifacts['failure_types'] = failure_types
    artifacts['taxonomy'] = taxonomy
    artifacts['dominant_failure'] = max(taxonomy.items(), key=lambda x: x[1])[0] if taxonomy else None

    telemetry['success'] = True
    telemetry['num_analyzed'] = len(failure_types)

    return NodeOutput(result=taxonomy, artifacts=artifacts, telemetry=telemetry)


def analyze_single_failure(pred: np.ndarray, target: np.ndarray, input_grid: np.ndarray = None) -> Dict[str, Any]:
    """
    Analyze a single prediction failure.

    Args:
        pred: Predicted grid
        target: Target grid
        input_grid: Original input grid (optional)

    Returns:
        Dictionary describing the failure
    """
    failure = {
        'type': 'unknown',
        'description': '',
        'severity': 0.0,
        'localizable': False,
    }

    # Perfect match
    iou = compute_iou(pred, target)
    if iou >= 0.99:
        failure['type'] = 'perfect'
        failure['severity'] = 0.0
        return failure

    # Shape mismatch
    if pred.shape != target.shape:
        failure['type'] = 'shape_mismatch'
        failure['description'] = f'Shape mismatch: predicted {pred.shape}, expected {target.shape}'
        failure['severity'] = 1.0
        failure['localizable'] = False
        return failure

    # Compute Hamming distance
    hamming = compute_hamming_distance(pred, target)
    total_cells = pred.size

    # Mostly correct (small errors)
    if iou >= 0.8:
        failure['type'] = 'partial_correct'
        failure['description'] = f'{hamming} cells incorrect out of {total_cells}'
        failure['severity'] = 1.0 - iou
        failure['localizable'] = True

        # Check if errors are localized
        diff_mask = pred != target
        if diff_mask.any():
            # Count connected components in the diff
            from scipy import ndimage
            labeled, num_features = ndimage.label(diff_mask)
            if num_features <= 3:  # Localized errors
                failure['description'] += f' (localized in {num_features} regions)'

        return failure

    # Color errors (shape preserved but colors wrong)
    if iou >= 0.5:
        failure['type'] = 'color_error'
        failure['description'] = f'Color errors in {hamming} cells'
        failure['severity'] = 1.0 - iou
        failure['localizable'] = True
        return failure

    # Placement errors (similar patterns but misaligned)
    if iou >= 0.3:
        failure['type'] = 'placement_error'
        failure['description'] = 'Objects may be misplaced or misaligned'
        failure['severity'] = 1.0 - iou
        failure['localizable'] = True
        return failure

    # Completely wrong
    failure['type'] = 'completely_wrong'
    failure['description'] = f'Very poor match (IoU: {iou:.2f})'
    failure['severity'] = 1.0
    failure['localizable'] = False

    return failure


# Node factory functions
def create_iou_critic_node() -> Node:
    """Create an IoU critic node."""
    return Node(
        name="iou_critic",
        func=iou_critic_func,
        input_type="predictions_and_targets",
        output_type="scores",
        deterministic=True,
        category="critic"
    )


def create_failure_analyzer_node() -> Node:
    """Create a failure analyzer node."""
    return Node(
        name="failure_analyzer",
        func=failure_analyzer_func,
        input_type="predictions_and_targets",
        output_type="failure_taxonomy",
        deterministic=True,
        category="critic"
    )
