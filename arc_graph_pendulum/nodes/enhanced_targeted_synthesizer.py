"""
Enhanced Targeted Synthesizer: Handles complex transformation types.
Supports pattern-based tiling, pattern extraction, and object operations.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes.targeted_synthesizer import TargetedSynthesizer as BaseTargetedSynthesizer
from core.node import Node, NodeOutput


class EnhancedTargetedSynthesizer(BaseTargetedSynthesizer):
    """
    Enhanced synthesizer that handles complex transformation rules.
    """

    def synthesize(self, rule: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Synthesize programs including complex transformations."""

        # Try base synthesizer first
        programs = []

        # Always include identity
        programs.append({
            'type': 'identity',
            'function': self.dsl.get('identity'),
            'description': 'Identity (baseline)',
            'confidence': 0.5,
            'operations': ['identity'],
        })

        rule_type = rule.get('rule_type', 'unknown')
        confidence = rule.get('confidence', 0.0)
        parameters = rule.get('parameters', {})

        # Handle new transformation types
        if rule_type == 'pattern_based_tiling':
            prog = self._synthesize_pattern_tiling(parameters, confidence)
            if prog:
                programs.insert(0, prog)

        elif rule_type == 'pattern_extraction':
            prog = self._synthesize_pattern_extraction(parameters, confidence)
            if prog:
                programs.insert(0, prog)

        elif rule_type == 'object_translate_all':
            prog = self._synthesize_object_translate(parameters, confidence)
            if prog:
                programs.insert(0, prog)

        elif rule_type == 'object_operations':
            # Per-object operations - more complex
            prog = self._synthesize_object_operations(parameters, confidence)
            if prog:
                programs.insert(0, prog)

        else:
            # Fall back to base synthesizer for standard transformations
            base_programs = super().synthesize(rule)
            # Merge, avoiding duplicate identity
            for prog in base_programs:
                if prog['type'] != 'identity':
                    programs.insert(0, prog)

        # Sort by confidence
        programs.sort(key=lambda p: p['confidence'], reverse=True)

        return programs[:5]

    def _synthesize_pattern_tiling(
        self,
        parameters: Dict[str, Any],
        confidence: float
    ) -> Optional[Dict]:
        """
        Synthesize program for pattern-based tiling.
        E.g., 3x3 → 9x9 where each input cell controls a 3x3 block.
        """
        scale_h = parameters.get('scale_h', 3)
        scale_w = parameters.get('scale_w', 3)

        def pattern_tiling_func(grid: np.ndarray) -> np.ndarray:
            """
            Pattern-based tiling: each input cell controls an output block.
            Non-zero cells → replicate pattern block
            Zero cells → fill with zeros
            """
            h, w = grid.shape

            # Find a pattern block (from first non-zero cell)
            pattern_block = None

            for i in range(h):
                for j in range(w):
                    if grid[i, j] != 0:
                        # Use this cell's "neighborhood" as pattern
                        # For now, use a simple tiling of the whole grid
                        pattern_block = grid.copy()
                        break
                if pattern_block is not None:
                    break

            if pattern_block is None:
                # All zeros
                return np.zeros((h * scale_h, w * scale_w), dtype=grid.dtype)

            # Create output
            output = np.zeros((h * scale_h, w * scale_w), dtype=grid.dtype)

            # For each cell in input
            for i in range(h):
                for j in range(w):
                    # Determine output block location
                    block_i = i * scale_h
                    block_j = j * scale_w

                    if grid[i, j] != 0:
                        # Fill with pattern
                        output[block_i:block_i+scale_h, block_j:block_j+scale_w] = \
                            pattern_block[:scale_h, :scale_w]
                    else:
                        # Fill with zeros (already done)
                        pass

            return output

        return {
            'type': 'pattern_based_tiling',
            'function': pattern_tiling_func,
            'description': f'Pattern-based tiling {scale_h}x{scale_w}',
            'confidence': confidence * 0.9,  # Slightly lower due to complexity
            'operations': ['pattern_tiling'],
        }

    def _synthesize_pattern_extraction(
        self,
        parameters: Dict[str, Any],
        confidence: float
    ) -> Optional[Dict]:
        """
        Synthesize program for pattern extraction.
        Extracts certain rows/columns and repeats them.
        """
        axis = parameters.get('axis', 'rows')
        row_mapping = parameters.get('row_mapping', [])
        color_mapping = parameters.get('color_mapping', {})
        extension_factor = parameters.get('extension_factor', 1.0)

        def pattern_extraction_func(grid: np.ndarray) -> np.ndarray:
            """Extract and extend pattern from grid."""
            h, w = grid.shape

            if axis == 'rows':
                # Extract rows based on mapping
                output_rows = []

                for row_idx in row_mapping:
                    if 0 <= row_idx < h:
                        row = grid[row_idx].copy()

                        # Apply color mapping
                        for from_color, to_color in color_mapping.items():
                            row[row == from_color] = to_color

                        output_rows.append(row)

                if output_rows:
                    output = np.array(output_rows)
                else:
                    output = grid.copy()

            else:  # columns
                # Similar logic for columns
                output_cols = []

                for col_idx in row_mapping:  # reuse row_mapping var
                    if 0 <= col_idx < w:
                        col = grid[:, col_idx].copy()

                        # Apply color mapping
                        for from_color, to_color in color_mapping.items():
                            col[col == from_color] = to_color

                        output_cols.append(col)

                if output_cols:
                    output = np.array(output_cols).T
                else:
                    output = grid.copy()

            return output

        return {
            'type': 'pattern_extraction',
            'function': pattern_extraction_func,
            'description': f'Pattern extraction along {axis} with color remap',
            'confidence': confidence * 0.85,
            'operations': ['pattern_extraction', 'color_remap'],
        }

    def _synthesize_object_translate(
        self,
        parameters: Dict[str, Any],
        confidence: float
    ) -> Optional[Dict]:
        """Synthesize program for uniform object translation."""
        offset = parameters.get('offset', (0, 0))
        dy, dx = offset

        def object_translate_func(grid: np.ndarray) -> np.ndarray:
            """Translate entire grid."""
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

        return {
            'type': 'object_translate_all',
            'function': object_translate_func,
            'description': f'Translate all objects by ({dy}, {dx})',
            'confidence': confidence,
            'operations': ['translate'],
        }

    def _synthesize_object_operations(
        self,
        parameters: Dict[str, Any],
        confidence: float
    ) -> Optional[Dict]:
        """Synthesize program for per-object operations (complex)."""
        # This is very complex - for now, return None
        # Would need to implement per-object transformation logic
        return None


def enhanced_targeted_synthesizer_func(data: Dict[str, Any]) -> NodeOutput:
    """Enhanced targeted synthesis function for node."""
    artifacts = {}
    telemetry = {'node_type': 'reasoner', 'subtype': 'enhanced_targeted_synthesizer'}

    rule = data.get('rule', None)

    if rule is None:
        telemetry['success'] = False
        return NodeOutput(result=[], artifacts=artifacts, telemetry=telemetry)

    synthesizer = EnhancedTargetedSynthesizer()
    programs = synthesizer.synthesize(rule)

    artifacts['programs'] = programs
    telemetry['num_programs'] = len(programs)
    telemetry['rule_type'] = rule.get('rule_type', 'unknown')
    telemetry['success'] = True

    return NodeOutput(result=programs, artifacts=artifacts, telemetry=telemetry)


def create_enhanced_targeted_synthesizer_node() -> Node:
    """Create an enhanced targeted synthesizer node."""
    return Node(
        name="enhanced_targeted_synthesizer",
        func=enhanced_targeted_synthesizer_func,
        input_type="rule",
        output_type="programs",
        deterministic=True,
        category="reasoner"
    )
