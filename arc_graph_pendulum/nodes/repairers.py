"""
Repair nodes for fixing localized failures.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Callable, Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.node import Node, NodeOutput
from utils.grid_utils import compute_iou


def placement_repairer_func(data: Dict[str, Any]) -> NodeOutput:
    """
    Repair placement/translation errors by searching for best offset.

    Args:
        data: Dictionary with 'program', 'predictions', 'targets', 'inputs'

    Returns:
        NodeOutput with repaired program
    """
    artifacts = {}
    telemetry = {'node_type': 'repairer', 'subtype': 'placement'}

    program = data.get('program')
    predictions = data.get('predictions', [])
    targets = data.get('targets', [])
    inputs = data.get('inputs', [])

    if not program or not predictions or not targets:
        telemetry['error'] = 'Missing required data'
        telemetry['success'] = False
        return NodeOutput(result=None, artifacts=artifacts, telemetry=telemetry)

    # Try to find best translation offset
    best_offset = (0, 0)
    best_score = 0.0

    # Search range for translations
    search_range = 5  # pixels in each direction

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            # Test this offset
            total_score = 0.0

            for pred, target in zip(predictions, targets):
                # Apply translation
                translated = translate_grid(pred, dy, dx, target.shape)
                score = compute_iou(translated, target)
                total_score += score

            avg_score = total_score / len(predictions)

            if avg_score > best_score:
                best_score = avg_score
                best_offset = (dy, dx)

    # Create repaired program if translation helps
    if best_offset != (0, 0) and best_score > 0.8:
        dy, dx = best_offset

        def repaired_func(grid):
            """Repaired function with translation."""
            result = program['function'](grid)
            # Apply translation
            if isinstance(result, np.ndarray):
                return translate_grid(result, dy, dx, result.shape)
            return result

        repaired_program = {
            'type': f"{program['type']}_translated",
            'function': repaired_func,
            'description': f"{program['description']} + translate({dy}, {dx})",
            'confidence': program.get('confidence', 0.5) * 0.95,
            'repair_type': 'placement',
            'offset': best_offset,
        }

        artifacts['repaired_program'] = repaired_program
        artifacts['offset'] = best_offset
        artifacts['improvement'] = best_score
        telemetry['success'] = True
        telemetry['repaired'] = True

        return NodeOutput(result=repaired_program, artifacts=artifacts, telemetry=telemetry)

    # No improvement
    telemetry['success'] = True
    telemetry['repaired'] = False
    return NodeOutput(result=program, artifacts=artifacts, telemetry=telemetry)


def color_repairer_func(data: Dict[str, Any]) -> NodeOutput:
    """
    Repair color mapping errors by learning color transformations.

    Args:
        data: Dictionary with 'program', 'predictions', 'targets'

    Returns:
        NodeOutput with repaired program
    """
    artifacts = {}
    telemetry = {'node_type': 'repairer', 'subtype': 'color'}

    program = data.get('program')
    predictions = data.get('predictions', [])
    targets = data.get('targets', [])

    if not program or not predictions or not targets:
        telemetry['error'] = 'Missing required data'
        telemetry['success'] = False
        return NodeOutput(result=None, artifacts=artifacts, telemetry=telemetry)

    # Learn color mapping from predictions to targets
    color_map = learn_color_mapping(predictions, targets)

    # Check if color mapping helps
    if color_map:
        # Test the mapping
        total_score = 0.0
        for pred, target in zip(predictions, targets):
            remapped = apply_color_map(pred, color_map)
            score = compute_iou(remapped, target)
            total_score += score

        avg_score = total_score / len(predictions)

        if avg_score > 0.8:
            # Create repaired program
            def repaired_func(grid):
                """Repaired function with color mapping."""
                result = program['function'](grid)
                if isinstance(result, np.ndarray):
                    return apply_color_map(result, color_map)
                return result

            repaired_program = {
                'type': f"{program['type']}_color_mapped",
                'function': repaired_func,
                'description': f"{program['description']} + color_remap",
                'confidence': program.get('confidence', 0.5) * 0.95,
                'repair_type': 'color',
                'color_map': color_map,
            }

            artifacts['repaired_program'] = repaired_program
            artifacts['color_map'] = color_map
            artifacts['improvement'] = avg_score
            telemetry['success'] = True
            telemetry['repaired'] = True

            return NodeOutput(result=repaired_program, artifacts=artifacts, telemetry=telemetry)

    # No improvement
    telemetry['success'] = True
    telemetry['repaired'] = False
    return NodeOutput(result=program, artifacts=artifacts, telemetry=telemetry)


def scale_repairer_func(data: Dict[str, Any]) -> NodeOutput:
    """
    Repair scale/size errors by trying different scaling factors.

    Args:
        data: Dictionary with 'program', 'predictions', 'targets'

    Returns:
        NodeOutput with repaired program
    """
    artifacts = {}
    telemetry = {'node_type': 'repairer', 'subtype': 'scale'}

    program = data.get('program')
    predictions = data.get('predictions', [])
    targets = data.get('targets', [])
    inputs = data.get('inputs', [])

    if not program or not predictions or not targets:
        telemetry['error'] = 'Missing required data'
        telemetry['success'] = False
        return NodeOutput(result=None, artifacts=artifacts, telemetry=telemetry)

    # Check if there's a consistent scale mismatch
    scale_factors = []
    for pred, target in zip(predictions, targets):
        if pred.shape[0] > 0 and pred.shape[1] > 0:
            scale_y = target.shape[0] / pred.shape[0]
            scale_x = target.shape[1] / pred.shape[1]
            scale_factors.append((scale_y, scale_x))

    if not scale_factors:
        telemetry['success'] = True
        telemetry['repaired'] = False
        return NodeOutput(result=program, artifacts=artifacts, telemetry=telemetry)

    # Check if scale is consistent
    avg_scale_y = np.mean([s[0] for s in scale_factors])
    avg_scale_x = np.mean([s[1] for s in scale_factors])
    var_scale = np.var([s[0] for s in scale_factors] + [s[1] for s in scale_factors])

    # If scale is consistent and not 1.0, try scaling
    if var_scale < 0.1 and (abs(avg_scale_y - 1.0) > 0.1 or abs(avg_scale_x - 1.0) > 0.1):
        # Try integer scales
        possible_scales = [(1, 1), (2, 2), (3, 3), (0.5, 0.5), (2, 1), (1, 2)]
        best_scale = (1, 1)
        best_score = 0.0

        for scale in possible_scales:
            total_score = 0.0
            for pred, target in zip(predictions, targets):
                scaled = scale_grid(pred, scale, target.shape)
                if scaled.shape == target.shape:
                    score = compute_iou(scaled, target)
                    total_score += score

            avg_score = total_score / len(predictions) if predictions else 0.0

            if avg_score > best_score:
                best_score = avg_score
                best_scale = scale

        if best_scale != (1, 1) and best_score > 0.8:
            # Create repaired program
            scale_y, scale_x = best_scale

            def repaired_func(grid):
                """Repaired function with scaling."""
                result = program['function'](grid)
                if isinstance(result, np.ndarray):
                    # Calculate target shape based on scale
                    target_shape = (int(result.shape[0] * scale_y), int(result.shape[1] * scale_x))
                    return scale_grid(result, best_scale, target_shape)
                return result

            repaired_program = {
                'type': f"{program['type']}_scaled",
                'function': repaired_func,
                'description': f"{program['description']} + scale({scale_y}, {scale_x})",
                'confidence': program.get('confidence', 0.5) * 0.95,
                'repair_type': 'scale',
                'scale': best_scale,
            }

            artifacts['repaired_program'] = repaired_program
            artifacts['scale'] = best_scale
            artifacts['improvement'] = best_score
            telemetry['success'] = True
            telemetry['repaired'] = True

            return NodeOutput(result=repaired_program, artifacts=artifacts, telemetry=telemetry)

    # No improvement
    telemetry['success'] = True
    telemetry['repaired'] = False
    return NodeOutput(result=program, artifacts=artifacts, telemetry=telemetry)


# Helper functions

def translate_grid(grid: np.ndarray, dy: int, dx: int, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Translate a grid by (dy, dx) offset.

    Args:
        grid: Input grid
        dy: Vertical offset
        dx: Horizontal offset
        target_shape: Target shape to fit into

    Returns:
        Translated grid
    """
    result = np.zeros(target_shape, dtype=grid.dtype)

    # Calculate source and destination bounds
    src_y1 = max(0, -dy)
    src_y2 = min(grid.shape[0], target_shape[0] - dy)
    src_x1 = max(0, -dx)
    src_x2 = min(grid.shape[1], target_shape[1] - dx)

    dst_y1 = max(0, dy)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x1 = max(0, dx)
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    if src_y2 > src_y1 and src_x2 > src_x1:
        result[dst_y1:dst_y2, dst_x1:dst_x2] = grid[src_y1:src_y2, src_x1:src_x2]

    return result


def learn_color_mapping(predictions: List[np.ndarray], targets: List[np.ndarray]) -> Optional[Dict[int, int]]:
    """
    Learn color mapping from predictions to targets.

    Args:
        predictions: List of predicted grids
        targets: List of target grids

    Returns:
        Dictionary mapping colors, or None
    """
    # Count color co-occurrences
    color_votes = {}  # (pred_color, target_color) -> count

    for pred, target in zip(predictions, targets):
        if pred.shape != target.shape:
            continue

        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pred_color = pred[i, j]
                target_color = target[i, j]

                key = (pred_color, target_color)
                color_votes[key] = color_votes.get(key, 0) + 1

    if not color_votes:
        return None

    # Build mapping by taking most common target color for each prediction color
    color_map = {}
    for pred_color in range(10):
        # Find most common target for this pred_color
        candidates = [(count, target_c) for (pred_c, target_c), count in color_votes.items() if pred_c == pred_color]

        if candidates:
            candidates.sort(reverse=True)
            _, best_target = candidates[0]

            # Only add if it's actually different
            if best_target != pred_color and len(candidates) > 0:
                color_map[pred_color] = best_target

    return color_map if color_map else None


def apply_color_map(grid: np.ndarray, color_map: Dict[int, int]) -> np.ndarray:
    """
    Apply color mapping to a grid.

    Args:
        grid: Input grid
        color_map: Color mapping dictionary

    Returns:
        Grid with colors remapped
    """
    result = grid.copy()

    for old_color, new_color in color_map.items():
        result[grid == old_color] = new_color

    return result


def scale_grid(grid: np.ndarray, scale: Tuple[float, float], target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Scale a grid by given factors.

    Args:
        grid: Input grid
        scale: (scale_y, scale_x)
        target_shape: Target shape

    Returns:
        Scaled grid
    """
    scale_y, scale_x = scale

    if scale_y == 1.0 and scale_x == 1.0:
        return grid

    # Simple nearest-neighbor scaling
    if scale_y >= 1.0 and scale_x >= 1.0:
        # Upscale by repetition
        result = np.repeat(grid, int(scale_y), axis=0)
        result = np.repeat(result, int(scale_x), axis=1)
    else:
        # Downscale by sampling
        step_y = int(1.0 / scale_y) if scale_y < 1.0 else 1
        step_x = int(1.0 / scale_x) if scale_x < 1.0 else 1
        result = grid[::step_y, ::step_x]

    # Crop or pad to target shape
    if result.shape[0] > target_shape[0]:
        result = result[:target_shape[0], :]
    elif result.shape[0] < target_shape[0]:
        padding = np.zeros((target_shape[0] - result.shape[0], result.shape[1]), dtype=grid.dtype)
        result = np.vstack([result, padding])

    if result.shape[1] > target_shape[1]:
        result = result[:, :target_shape[1]]
    elif result.shape[1] < target_shape[1]:
        padding = np.zeros((result.shape[0], target_shape[1] - result.shape[1]), dtype=grid.dtype)
        result = np.hstack([result, padding])

    return result


# Node factory functions

def create_placement_repairer_node() -> Node:
    """Create a placement repairer node."""
    return Node(
        name="placement_repairer",
        func=placement_repairer_func,
        input_type="program_and_predictions",
        output_type="repaired_program",
        deterministic=False,  # Depends on search
        category="repairer"
    )


def create_color_repairer_node() -> Node:
    """Create a color repairer node."""
    return Node(
        name="color_repairer",
        func=color_repairer_func,
        input_type="program_and_predictions",
        output_type="repaired_program",
        deterministic=True,
        category="repairer"
    )


def create_scale_repairer_node() -> Node:
    """Create a scale repairer node."""
    return Node(
        name="scale_repairer",
        func=scale_repairer_func,
        input_type="program_and_predictions",
        output_type="repaired_program",
        deterministic=True,
        category="repairer"
    )
