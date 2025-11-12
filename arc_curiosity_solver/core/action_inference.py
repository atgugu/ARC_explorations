"""
Action Inference from Training Data

Analyzes training examples to detect which transformations occur,
enabling the solver to focus on relevant actions rather than trying all possibilities.

This is Phase 6 Priority 1: Action Learning.
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from scipy import ndimage
from collections import Counter

from ..core.object_reasoning import ArcObject, ObjectDetector


class ActionInference:
    """
    Infers which actions/transformations occur in training examples.

    This allows the solver to focus hypothesis generation on actions
    that are actually relevant to the task, rather than trying all
    possible actions blindly.
    """

    def __init__(self):
        self.detector = ObjectDetector()

    def analyze_training_pairs(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, any]:
        """
        Analyze all training pairs to detect transformations.

        Returns dict with detected actions and their confidence:
        {
            'rotations': [90, 180, 270],  # Detected rotation angles
            'reflections': ['horizontal', 'vertical'],  # Detected reflections
            'color_swaps': [(1, 2), (3, 4)],  # Detected color mappings
            'size_changes': ['scale_up', 'scale_down'],
            'position_changes': ['move', 'center', 'edge'],
            'extensions': ['top', 'bottom', 'left', 'right'],
            'replications': True/False,
            'removals': True/False,
            'confidence': {'rotations': 0.0-1.0, ...}  # Confidence scores
        }
        """
        detected = {
            'rotations': set(),
            'reflections': set(),
            'color_swaps': [],
            'size_changes': set(),
            'position_changes': set(),
            'extensions': set(),
            'replications': False,
            'removals': False,
            'confidence': {}
        }

        # Count detections per pair for confidence calculation
        rotation_counts = {90: 0, 180: 0, 270: 0}
        reflection_counts = {'horizontal': 0, 'vertical': 0}
        total_pairs = len(train_pairs)

        for inp, out in train_pairs:
            # Grid-level detection (works for whole-grid transformations)
            grid_rotations = self._detect_rotation(inp, out)
            if grid_rotations:
                detected['rotations'].update(grid_rotations)
                for angle in grid_rotations:
                    rotation_counts[angle] += 1

            grid_reflections = self._detect_reflection(inp, out)
            if grid_reflections:
                detected['reflections'].update(grid_reflections)
                for refl in grid_reflections:
                    reflection_counts[refl] += 1

            # Object-level detection (works for object transformations within grids)
            inp_objs = self.detector.detect_objects(inp)
            out_objs = self.detector.detect_objects(out)

            obj_rotations = self._detect_object_rotation(inp_objs, out_objs)
            if obj_rotations:
                detected['rotations'].update(obj_rotations)
                for angle in obj_rotations:
                    rotation_counts[angle] += 1

            obj_reflections = self._detect_object_reflection(inp_objs, out_objs)
            if obj_reflections:
                detected['reflections'].update(obj_reflections)
                for refl in obj_reflections:
                    reflection_counts[refl] += 1

            # Detect color transformations
            color_swaps = self._detect_color_mapping(inp, out)
            if color_swaps:
                detected['color_swaps'].extend(color_swaps)

            # Size changes
            if self._detect_size_change(inp_objs, out_objs):
                detected['size_changes'].add('scale')

            # Position changes
            pos_changes = self._detect_position_change(inp_objs, out_objs)
            detected['position_changes'].update(pos_changes)

            # Extensions
            extensions = self._detect_extension(inp, out, inp_objs, out_objs)
            detected['extensions'].update(extensions)

            # Replications
            if self._detect_replication(inp_objs, out_objs):
                detected['replications'] = True

            # Removals
            if len(out_objs) < len(inp_objs):
                detected['removals'] = True

        # Calculate confidence scores based on detection frequency
        if detected['rotations']:
            max_count = max(rotation_counts[angle] for angle in detected['rotations'])
            detected['confidence']['rotations'] = max_count / total_pairs

        if detected['reflections']:
            max_count = max(reflection_counts[refl] for refl in detected['reflections'])
            detected['confidence']['reflections'] = max_count / total_pairs

        # Other actions get confidence 0.5 if detected (moderate confidence)
        for key in ['extensions', 'position_changes', 'color_swaps']:
            if detected[key]:
                detected['confidence'][key] = 0.5

        if detected['replications']:
            detected['confidence']['replications'] = 0.5

        return detected

    def _detect_rotation(self, inp: np.ndarray, out: np.ndarray) -> Set[int]:
        """Detect if output is a rotation of input."""
        detected_angles = set()

        # Try different rotation angles
        for angle, k in [(90, -1), (180, 2), (270, 1)]:
            rotated = np.rot90(inp, k=k)

            # Check if rotated matches output
            if rotated.shape == out.shape:
                similarity = (rotated == out).mean()
                if similarity > 0.8:  # 80% match
                    detected_angles.add(angle)

        return detected_angles

    def _detect_reflection(self, inp: np.ndarray, out: np.ndarray) -> Set[str]:
        """Detect if output is a reflection of input."""
        detected_reflections = set()

        # Horizontal reflection (left-right flip)
        reflected_h = np.fliplr(inp)
        if reflected_h.shape == out.shape:
            similarity = (reflected_h == out).mean()
            if similarity > 0.8:
                detected_reflections.add('horizontal')

        # Vertical reflection (up-down flip)
        reflected_v = np.flipud(inp)
        if reflected_v.shape == out.shape:
            similarity = (reflected_v == out).mean()
            if similarity > 0.8:
                detected_reflections.add('vertical')

        return detected_reflections

    def _detect_object_rotation(self, inp_objs: List[ArcObject], out_objs: List[ArcObject]) -> Set[int]:
        """
        Detect if objects were rotated.

        Compares individual objects between input and output to find rotations.
        This catches object-level transformations that grid-level detection misses.
        """
        detected_angles = set()

        if not inp_objs or not out_objs:
            return detected_angles

        # Try to match input objects to output objects
        for inp_obj in inp_objs:
            inp_grid = inp_obj.grid

            for out_obj in out_objs:
                # Skip if different colors (unlikely same object)
                if inp_obj.dominant_color != out_obj.dominant_color:
                    continue

                out_grid = out_obj.grid

                # Try different rotations
                for angle, k in [(90, -1), (180, 2), (270, 1)]:
                    try:
                        rotated = np.rot90(inp_grid, k=k)

                        # Check if rotated matches output object
                        if rotated.shape == out_grid.shape:
                            # More lenient threshold for objects (may have noise)
                            similarity = (rotated == out_grid).mean()
                            if similarity > 0.6:  # 60% match for objects
                                detected_angles.add(angle)
                    except:
                        pass

        return detected_angles

    def _detect_object_reflection(self, inp_objs: List[ArcObject], out_objs: List[ArcObject]) -> Set[str]:
        """
        Detect if objects were reflected.

        Compares individual objects between input and output to find reflections.
        This catches object-level transformations that grid-level detection misses.
        """
        detected_reflections = set()

        if not inp_objs or not out_objs:
            return detected_reflections

        # Try to match input objects to output objects
        for inp_obj in inp_objs:
            inp_grid = inp_obj.grid

            for out_obj in out_objs:
                # Skip if different colors (unlikely same object)
                if inp_obj.dominant_color != out_obj.dominant_color:
                    continue

                out_grid = out_obj.grid

                # Try horizontal reflection
                try:
                    reflected_h = np.fliplr(inp_grid)
                    if reflected_h.shape == out_grid.shape:
                        similarity = (reflected_h == out_grid).mean()
                        if similarity > 0.6:  # 60% match for objects
                            detected_reflections.add('horizontal')
                except:
                    pass

                # Try vertical reflection
                try:
                    reflected_v = np.flipud(inp_grid)
                    if reflected_v.shape == out_grid.shape:
                        similarity = (reflected_v == out_grid).mean()
                        if similarity > 0.6:  # 60% match for objects
                            detected_reflections.add('vertical')
                except:
                    pass

        return detected_reflections

    def _detect_color_mapping(self, inp: np.ndarray, out: np.ndarray) -> List[Tuple[int, int]]:
        """Detect color swaps/mappings between input and output."""
        if inp.shape != out.shape:
            return []

        color_mappings = []

        # Get unique colors
        inp_colors = set(np.unique(inp))
        out_colors = set(np.unique(out))

        # If colors are different, there's a mapping
        if inp_colors != out_colors:
            # Try to find which colors map to which
            for inp_color in inp_colors:
                if inp_color == 0:  # Skip background
                    continue

                # Where does this input color appear?
                inp_mask = (inp == inp_color)

                # What colors appear in output at those positions?
                out_at_inp_positions = out[inp_mask]
                if len(out_at_inp_positions) > 0:
                    # Most common color at those positions
                    out_color = Counter(out_at_inp_positions).most_common(1)[0][0]

                    if out_color != inp_color:
                        color_mappings.append((inp_color, out_color))

        return color_mappings

    def _detect_size_change(self, inp_objs: List[ArcObject], out_objs: List[ArcObject]) -> bool:
        """Detect if objects changed size."""
        if not inp_objs or not out_objs:
            return False

        # Compare average sizes
        inp_avg_size = np.mean([obj.size for obj in inp_objs])
        out_avg_size = np.mean([obj.size for obj in out_objs])

        # If sizes differ significantly, there's a size change
        if abs(out_avg_size - inp_avg_size) / max(inp_avg_size, 1) > 0.3:
            return True

        return False

    def _detect_position_change(self, inp_objs: List[ArcObject], out_objs: List[ArcObject]) -> Set[str]:
        """Detect types of position changes."""
        changes = set()

        if not inp_objs or not out_objs:
            return changes

        # Check if objects moved
        for inp_obj in inp_objs:
            for out_obj in out_objs:
                # Check if they're the same object (similar size and color)
                if (abs(inp_obj.size - out_obj.size) < 5 and
                    inp_obj.dominant_color == out_obj.dominant_color):

                    # Check position change
                    inp_y, inp_x = inp_obj.position
                    out_y, out_x = out_obj.position

                    if (inp_y, inp_x) != (out_y, out_x):
                        changes.add('move')

                        # Check if moved to center (roughly)
                        # (This is approximate - would need grid size)

                        # Check if moved to edge
                        if out_y == 0 or out_x == 0:
                            changes.add('edge')

        return changes

    def _detect_extension(self, inp: np.ndarray, out: np.ndarray,
                         inp_objs: List[ArcObject], out_objs: List[ArcObject]) -> Set[str]:
        """Detect if objects were extended in some direction."""
        extensions = set()

        if inp.shape != out.shape:
            return extensions

        for inp_obj in inp_objs:
            for out_obj in out_objs:
                # Check if same color
                if inp_obj.dominant_color != out_obj.dominant_color:
                    continue

                inp_y1, inp_x1, inp_y2, inp_x2 = inp_obj.bbox
                out_y1, out_x1, out_y2, out_x2 = out_obj.bbox

                # Check if bbox extended
                if out_y1 < inp_y1:
                    extensions.add('top')
                if out_y2 > inp_y2:
                    extensions.add('bottom')
                if out_x1 < inp_x1:
                    extensions.add('left')
                if out_x2 > inp_x2:
                    extensions.add('right')

        return extensions

    def _detect_replication(self, inp_objs: List[ArcObject], out_objs: List[ArcObject]) -> bool:
        """Detect if objects were replicated."""
        if len(out_objs) <= len(inp_objs):
            return False

        # Simple heuristic: if output has more objects of same color/size, likely replication
        inp_signatures = [(obj.dominant_color, obj.size // 10) for obj in inp_objs]
        out_signatures = [(obj.dominant_color, obj.size // 10) for obj in out_objs]

        inp_counts = Counter(inp_signatures)
        out_counts = Counter(out_signatures)

        # If any signature appears more in output, it's replication
        for sig in inp_counts:
            if out_counts.get(sig, 0) > inp_counts[sig]:
                return True

        return False


class ActionFocusedGenerator:
    """
    Generates hypotheses focused on detected actions.

    Uses ActionInference to determine which actions to prioritize,
    reducing wasted hypothesis generation.
    """

    def __init__(self, action_inference: ActionInference):
        self.action_inference = action_inference

    def should_generate_rotation(self, detected_actions: Dict) -> List[int]:
        """Returns which rotation angles to try (if any)."""
        if detected_actions.get('rotations'):
            return list(detected_actions['rotations'])
        return []  # Don't generate rotation hypotheses if not detected

    def should_generate_reflection(self, detected_actions: Dict) -> List[str]:
        """Returns which reflections to try (if any)."""
        if detected_actions.get('reflections'):
            return list(detected_actions['reflections'])
        return []

    def should_generate_color_swap(self, detected_actions: Dict) -> bool:
        """Returns whether to generate color swap hypotheses."""
        return bool(detected_actions.get('color_swaps'))

    def should_generate_extension(self, detected_actions: Dict) -> List[str]:
        """Returns which extension directions to try (if any)."""
        if detected_actions.get('extensions'):
            return list(detected_actions['extensions'])
        return []

    def should_generate_replication(self, detected_actions: Dict) -> bool:
        """Returns whether to generate replication hypotheses."""
        return detected_actions.get('replications', False)

    def get_priority_actions(self, detected_actions: Dict) -> List[str]:
        """
        Returns list of action types to prioritize, in order.

        Based on confidence and detection in training data.
        """
        priority = []

        # Add detected actions with high confidence
        if detected_actions.get('rotations'):
            priority.append('rotation')

        if detected_actions.get('reflections'):
            priority.append('reflection')

        if detected_actions.get('color_swaps'):
            priority.append('color_swap')

        if detected_actions.get('extensions'):
            priority.append('extension')

        if detected_actions.get('replications'):
            priority.append('replication')

        if detected_actions.get('position_changes'):
            priority.append('movement')

        # Always try recoloring as fallback (most common)
        if 'recolor' not in priority:
            priority.append('recolor')

        return priority
