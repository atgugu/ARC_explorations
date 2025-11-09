"""
Advanced program synthesizer with feature-driven generation and composition.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.node import Node, NodeOutput
from core.dsl import DSLRegistry, Operation, CompositeOperation


class AdvancedProgramSynthesizer:
    """
    Advanced synthesizer that generates diverse programs using:
    1. Feature-driven synthesis
    2. Compositional program generation
    3. Smart filtering and ranking
    """

    def __init__(self):
        self.dsl = DSLRegistry()
        self.max_programs = 100

    def synthesize_programs(
        self,
        task_data: Dict[str, Any],
        facts: Dict[str, Any],
        hypotheses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Synthesize programs using features and hypotheses.

        Args:
            task_data: Task data with train/test examples
            facts: Extracted facts from feature nodes
            hypotheses: Generated hypotheses

        Returns:
            List of program dictionaries
        """
        programs = []

        # 1. Generate single-operation programs
        programs.extend(self._generate_single_op_programs())

        # 2. Feature-driven synthesis
        programs.extend(self._symmetry_driven_synthesis(facts))
        programs.extend(self._color_driven_synthesis(facts))
        programs.extend(self._shape_driven_synthesis(facts))
        programs.extend(self._pattern_driven_synthesis(facts))
        programs.extend(self._object_driven_synthesis(facts))

        # 3. Hypothesis-driven synthesis
        programs.extend(self._hypothesis_driven_synthesis(hypotheses))

        # 4. Generate 2-step compositions
        programs.extend(self._generate_compositions(programs, depth=2, max_count=30))

        # 5. Generate targeted 3-step compositions
        programs.extend(self._generate_targeted_compositions(facts))

        # 6. Filter duplicates and rank
        programs = self._filter_and_rank(programs, task_data)

        # 7. Limit to top programs
        return programs[:self.max_programs]

    def _generate_single_op_programs(self) -> List[Dict[str, Any]]:
        """Generate programs for each single DSL operation."""
        programs = []

        for op in self.dsl.get_all():
            programs.append({
                'type': op.name,
                'function': op,
                'description': op.description,
                'confidence': 0.3,  # Base confidence
                'operations': [op.name],
            })

        return programs

    def _symmetry_driven_synthesis(self, facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate programs based on detected symmetries."""
        programs = []

        symmetries = facts.get('symmetries', [])
        if not symmetries:
            return programs

        # Check for consistent symmetries
        vertical_count = sum(s.get('output_vertical', False) for s in symmetries)
        horizontal_count = sum(s.get('output_horizontal', False) for s in symmetries)
        diagonal_count = sum(s.get('output_diagonal', False) for s in symmetries)

        total = len(symmetries)

        # If most examples have vertical symmetry
        if vertical_count >= total * 0.6:
            programs.append({
                'type': 'flip_h',
                'function': self.dsl.get('flip_h'),
                'description': 'Flip horizontally (vertical symmetry detected)',
                'confidence': 0.8,
                'operations': ['flip_h'],
            })

        # If most examples have horizontal symmetry
        if horizontal_count >= total * 0.6:
            programs.append({
                'type': 'flip_v',
                'function': self.dsl.get('flip_v'),
                'description': 'Flip vertically (horizontal symmetry detected)',
                'confidence': 0.8,
                'operations': ['flip_v'],
            })

        # If diagonal symmetry, try transpose
        if diagonal_count >= total * 0.6:
            programs.append({
                'type': 'transpose',
                'function': self.dsl.get('transpose'),
                'description': 'Transpose (diagonal symmetry detected)',
                'confidence': 0.8,
                'operations': ['transpose'],
            })

        return programs

    def _color_driven_synthesis(self, facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate programs based on color transformations."""
        programs = []

        # Check for color changes
        avg_changes = facts.get('avg_color_changes', None)
        if avg_changes is None:
            return programs

        # Detect color patterns
        significant_changes = np.abs(avg_changes) > 0.5

        if significant_changes.any():
            # Try color increment/decrement
            if np.sum(avg_changes > 0) > np.sum(avg_changes < 0):
                programs.append({
                    'type': 'increment',
                    'function': self.dsl.get('increment'),
                    'description': 'Increment colors (positive changes detected)',
                    'confidence': 0.6,
                    'operations': ['increment'],
                })
            else:
                programs.append({
                    'type': 'decrement',
                    'function': self.dsl.get('decrement'),
                    'description': 'Decrement colors (negative changes detected)',
                    'confidence': 0.6,
                    'operations': ['decrement'],
                })

        # Try color swaps
        programs.append({
            'type': 'swap_01',
            'function': self.dsl.get('swap_01'),
            'description': 'Swap colors 0↔1',
            'confidence': 0.4,
            'operations': ['swap_01'],
        })

        programs.append({
            'type': 'swap_12',
            'function': self.dsl.get('swap_12'),
            'description': 'Swap colors 1↔2',
            'confidence': 0.4,
            'operations': ['swap_12'],
        })

        return programs

    def _shape_driven_synthesis(self, facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate programs based on shape transformations."""
        programs = []

        shapes = facts.get('shapes', [])
        if not shapes:
            return programs

        # Check size ratios
        ratios = [s.get('size_ratio', 1.0) for s in shapes]
        avg_ratio = np.mean(ratios)

        # If output is typically 2x larger
        if 1.8 < avg_ratio < 2.2:
            programs.append({
                'type': 'scale_2x',
                'function': self.dsl.get('scale_2x'),
                'description': 'Scale 2x (2x size ratio detected)',
                'confidence': 0.7,
                'operations': ['scale_2x'],
            })

        # If output is typically 3x larger
        if 2.8 < avg_ratio < 3.2:
            programs.append({
                'type': 'scale_3x',
                'function': self.dsl.get('scale_3x'),
                'description': 'Scale 3x (3x size ratio detected)',
                'confidence': 0.7,
                'operations': ['scale_3x'],
            })

        # If output is typically 4x larger (2x2 tiling)
        if 3.8 < avg_ratio < 4.2:
            programs.append({
                'type': 'tile_2x2',
                'function': self.dsl.get('tile_2x2'),
                'description': 'Tile 2x2 (4x size ratio detected)',
                'confidence': 0.7,
                'operations': ['tile_2x2'],
            })

        # If output is typically smaller (downsampling)
        if 0.4 < avg_ratio < 0.6:
            programs.append({
                'type': 'downsample_2x',
                'function': self.dsl.get('downsample_2x'),
                'description': 'Downsample 2x (0.5x size ratio detected)',
                'confidence': 0.7,
                'operations': ['downsample_2x'],
            })

        return programs

    def _pattern_driven_synthesis(self, facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate programs based on detected patterns."""
        programs = []

        is_periodic = facts.get('is_periodic_task', False)

        if is_periodic:
            programs.append({
                'type': 'tile_2x2',
                'function': self.dsl.get('tile_2x2'),
                'description': 'Tile 2x2 (periodicity detected)',
                'confidence': 0.7,
                'operations': ['tile_2x2'],
            })

            programs.append({
                'type': 'tile_3x3',
                'function': self.dsl.get('tile_3x3'),
                'description': 'Tile 3x3 (periodicity detected)',
                'confidence': 0.6,
                'operations': ['tile_3x3'],
            })

        return programs

    def _object_driven_synthesis(self, facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate programs based on detected objects."""
        programs = []

        avg_input_objects = facts.get('avg_input_objects', 0)
        avg_output_objects = facts.get('avg_output_objects', 0)

        # If objects are being filtered/extracted
        if avg_output_objects < avg_input_objects * 0.8:
            programs.append({
                'type': 'largest_object',
                'function': self.dsl.get('largest_object'),
                'description': 'Extract largest object (object filtering detected)',
                'confidence': 0.6,
                'operations': ['largest_object'],
            })

        return programs

    def _hypothesis_driven_synthesis(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate programs based on hypotheses."""
        programs = []

        for hyp in hypotheses:
            hyp_type = hyp.get('type', 'unknown')

            # Map hypotheses to DSL operations
            if hyp_type == 'mirror_vertical':
                op = self.dsl.get('flip_h')
                if op:
                    programs.append({
                        'type': 'flip_h',
                        'function': op,
                        'description': f"Flip horizontal (hypothesis: {hyp['description']})",
                        'confidence': hyp.get('confidence', 0.5),
                        'operations': ['flip_h'],
                    })

            elif hyp_type == 'mirror_horizontal':
                op = self.dsl.get('flip_v')
                if op:
                    programs.append({
                        'type': 'flip_v',
                        'function': op,
                        'description': f"Flip vertical (hypothesis: {hyp['description']})",
                        'confidence': hyp.get('confidence', 0.5),
                        'operations': ['flip_v'],
                    })

        return programs

    def _generate_compositions(self, base_programs: List[Dict], depth: int, max_count: int) -> List[Dict[str, Any]]:
        """Generate compositional programs."""
        if depth < 2:
            return []

        compositions = []

        # Smart composition: only compose compatible operations
        for i, prog1 in enumerate(base_programs[:20]):  # Limit base set
            for prog2 in base_programs[i+1:i+6]:  # Limit combinations
                if self._are_compatible(prog1, prog2):
                    # Create composite
                    ops = prog1['operations'] + prog2['operations']
                    composite_func = self._create_composite_function(
                        prog1['function'],
                        prog2['function']
                    )

                    compositions.append({
                        'type': f"{prog1['type']}_{prog2['type']}",
                        'function': composite_func,
                        'description': f"{prog1['description']} then {prog2['description']}",
                        'confidence': min(prog1['confidence'], prog2['confidence']) * 0.9,
                        'operations': ops,
                    })

                    if len(compositions) >= max_count:
                        return compositions

        return compositions

    def _generate_targeted_compositions(self, facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific 3-step compositions based on features."""
        compositions = []

        # Example: Flip then translate then rotate
        # (These are common patterns in ARC)

        # Pattern 1: Flip + Rotate
        flip_h = self.dsl.get('flip_h')
        rotate_90 = self.dsl.get('rotate_90')
        if flip_h and rotate_90:
            composite = self._create_composite_function(flip_h, rotate_90)
            compositions.append({
                'type': 'flip_h_rotate_90',
                'function': composite,
                'description': 'Flip horizontal then rotate 90°',
                'confidence': 0.4,
                'operations': ['flip_h', 'rotate_90'],
            })

        # Pattern 2: Transpose + Flip
        transpose = self.dsl.get('transpose')
        if transpose and flip_h:
            composite = self._create_composite_function(transpose, flip_h)
            compositions.append({
                'type': 'transpose_flip_h',
                'function': composite,
                'description': 'Transpose then flip horizontal',
                'confidence': 0.4,
                'operations': ['transpose', 'flip_h'],
            })

        return compositions

    def _are_compatible(self, prog1: Dict, prog2: Dict) -> bool:
        """Check if two programs are compatible for composition."""
        # Avoid composing identity with anything
        if prog1['type'] == 'identity' or prog2['type'] == 'identity':
            return False

        # Avoid composing same operation
        if prog1['type'] == prog2['type']:
            return False

        # Avoid too many translations
        if 'translate' in prog1['type'] and 'translate' in prog2['type']:
            return False

        return True

    def _create_composite_function(self, func1: callable, func2: callable) -> callable:
        """Create a composite function."""
        def composite(grid):
            result = func1(grid)
            result = func2(result)
            return result

        return composite

    def _filter_and_rank(self, programs: List[Dict], task_data: Dict) -> List[Dict]:
        """Filter duplicates and rank programs."""
        # Remove duplicates by type
        seen_types = set()
        unique_programs = []

        for prog in programs:
            if prog['type'] not in seen_types:
                seen_types.add(prog['type'])
                unique_programs.append(prog)

        # Sort by confidence
        unique_programs.sort(key=lambda p: p['confidence'], reverse=True)

        return unique_programs


def advanced_program_synthesizer_func(data: Dict[str, Any]) -> NodeOutput:
    """
    Advanced program synthesis function for node.

    Args:
        data: Dictionary with 'hypotheses', 'facts', 'task_data'

    Returns:
        NodeOutput with synthesized programs
    """
    artifacts = {}
    telemetry = {'node_type': 'reasoner', 'subtype': 'advanced_program_synthesizer'}

    hypotheses = data.get('hypotheses', [])
    facts = data.get('facts', {})
    task_data = data.get('task_data', {})

    synthesizer = AdvancedProgramSynthesizer()
    programs = synthesizer.synthesize_programs(task_data, facts, hypotheses)

    artifacts['programs'] = programs
    telemetry['num_programs'] = len(programs)
    telemetry['success'] = True

    return NodeOutput(result=programs, artifacts=artifacts, telemetry=telemetry)


def create_advanced_program_synthesizer_node() -> Node:
    """Create an advanced program synthesizer node."""
    return Node(
        name="advanced_program_synthesizer",
        func=advanced_program_synthesizer_func,
        input_type="hypotheses_and_facts",
        output_type="programs",
        deterministic=True,
        category="reasoner"
    )
