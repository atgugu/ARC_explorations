"""
Conditional ARC Curiosity Solver

Integrates conditional transformations (IF-THEN-ELSE), spatial predicates,
and looping constructs to break through the 1% expressiveness barrier.

Expected improvement: +8-15% solve rate (from 1% to 10-16%)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from collections import Counter

from .solver_diverse import DiverseARCCuriositySolver
from .core.conditional_pattern_inference import (
    ConditionalPatternAnalyzer,
    ConditionalHypothesisGenerator,
    EnhancedPatternInference
)
from .core.improved_conditional_inference import ImprovedConditionalPatternAnalyzer
from .transformations.conditional_transforms import (
    ConditionalTransform,
    ConditionLibrary,
    ActionLibrary
)
from .transformations.nested_conditionals import (
    CompositeCondition,
    create_and_condition,
    create_or_condition,
    create_not_condition
)
from .transformations.multi_stage_compositions import (
    ConditionalPipeline,
    ConditionalStage,
    ConditionalLoop,
    PipelineBuilder
)
from .transformations.arc_primitives import Transform
from .core.object_reasoning import ObjectDetector


class ConditionalARCCuriositySolver(DiverseARCCuriositySolver):
    """
    Enhanced solver with conditional transformation capabilities.

    New capabilities:
    1. IF-THEN-ELSE logic based on object properties
    2. Spatial predicates (near_edge, touching, center, etc.)
    3. Conditional pattern inference from examples
    4. Property-dependent transformations
    5. Looping constructs (FOR each object)
    6. Multi-stage compositions

    Architecture:
    - Inherits curiosity-driven exploration from base solver
    - Inherits diversity mechanisms from DiverseARCCuriositySolver
    - Adds conditional pattern detection and generation
    - Prioritizes conditional hypotheses (higher expressiveness)
    """

    def __init__(self):
        super().__init__()

        # Conditional transformation components (IMPROVED)
        self.conditional_analyzer = ImprovedConditionalPatternAnalyzer()  # NEW: Improved version
        self.conditional_generator = ConditionalHypothesisGenerator()
        self.enhanced_inference = EnhancedPatternInference()

        # Libraries
        self.condition_lib = ConditionLibrary()
        self.action_lib = ActionLibrary()

        # Configuration
        self.use_conditionals = True
        self.use_spatial_predicates = True
        self.use_validated_conditionals = True  # NEW: Use validation
        self.use_nested_conditionals = True  # PHASE 3: AND/OR/NOT logic
        self.use_multi_stage = True  # PHASE 3: Sequential pipelines
        self.conditional_priority_boost = 1.5  # Boost confidence of conditional hypotheses
        self.nested_priority_boost = 3.0  # Even higher boost for nested (more expressive)

        # Phase 3 components
        self.pipeline_builder = PipelineBuilder()

    def _generate_hypotheses(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                            test_input: np.ndarray) -> List:
        """
        Generate hypotheses using conditional pattern inference.

        Priority order:
        1. Conditional patterns (IF-THEN-ELSE) - HIGHEST priority
        2. Unconditional patterns (from parent class)
        3. Diversity variations
        4. Generic transforms
        """
        from .belief_dynamics.belief_space import Hypothesis

        hypotheses = []

        if not train_pairs:
            return hypotheses

        # === CONDITIONAL PATTERN DETECTION (IMPROVED!) ===
        if self.use_conditionals and self.use_validated_conditionals:
            try:
                # Use improved analyzer with validation
                detected_conditionals = self.conditional_analyzer.analyze_training_pairs(train_pairs)

                for conditional in detected_conditionals:
                    # Create transform from conditional
                    def make_transform(ct=conditional):
                        def transform_fn(grid: np.ndarray) -> np.ndarray:
                            return ct.apply(grid, objects=None)
                        return transform_fn

                    transform_obj = Transform(
                        name=f"validated_conditional_{len(hypotheses)}",
                        function=make_transform(),
                        parameters={'variant': 'conditional', 'description': conditional.description},
                        category='conditional'
                    )

                    # Use validated confidence (already tested on training)
                    boosted_confidence = min(1.0, conditional.confidence * self.conditional_priority_boost)

                    hyp = Hypothesis(
                        program=transform_obj,
                        name=f"conditional_{len(hypotheses)}",
                        parameters={'description': conditional.description, 'variant': 'conditional'},
                        activation=boosted_confidence
                    )
                    hypotheses.append(hyp)

                    if len(hypotheses) >= 15:  # Reserve space for other hypothesis types
                        break

            except Exception as e:
                # Conditional detection failed, fall back to unconditional
                pass
        elif self.use_conditionals:
            # Fallback to old generator
            try:
                conditional_hyps = self.conditional_generator.generate_from_training(train_pairs)

                for transform_obj, description, confidence in conditional_hyps:
                    boosted_confidence = min(1.0, confidence * self.conditional_priority_boost)

                    hyp = Hypothesis(
                        program=transform_obj,
                        name=f"conditional_{len(hypotheses)}",
                        parameters={'description': description, 'variant': 'conditional'},
                        activation=boosted_confidence
                    )
                    hypotheses.append(hyp)

                    if len(hypotheses) >= 15:
                        break

            except Exception as e:
                pass

        # === PHASE 3: NESTED CONDITIONALS (AND/OR/NOT) - HIGH PRIORITY ===
        if self.use_nested_conditionals:
            try:
                nested_hyps = self._generate_nested_conditionals(train_pairs, test_input)
                for transform_obj, description, confidence in nested_hyps:
                    # NO CAPPING - let boost push these to top
                    boosted_confidence = confidence * self.nested_priority_boost
                    hyp = Hypothesis(
                        program=transform_obj,
                        name=f"nested_{len(hypotheses)}",
                        parameters={'description': description, 'variant': 'nested'},
                        activation=boosted_confidence
                    )
                    hypotheses.append(hyp)
            except Exception as e:
                pass

        # === PHASE 3: MULTI-STAGE PIPELINES - HIGH PRIORITY ===
        if self.use_multi_stage:
            try:
                pipeline_hyps = self._generate_multi_stage_pipelines(train_pairs, test_input)
                for transform_obj, description, confidence in pipeline_hyps:
                    # NO CAPPING - let boost push these to top
                    boosted_confidence = confidence * self.nested_priority_boost
                    hyp = Hypothesis(
                        program=transform_obj,
                        name=f"pipeline_{len(hypotheses)}",
                        parameters={'description': description, 'variant': 'pipeline'},
                        activation=boosted_confidence
                    )
                    hypotheses.append(hyp)
            except Exception as e:
                pass

        # === SPATIAL PREDICATE VARIATIONS ===
        # Generate variations of detected patterns using spatial conditions
        if self.use_spatial_predicates:
            spatial_hyps = self._generate_spatial_variations(train_pairs, test_input)
            for transform_obj, description, confidence in spatial_hyps:
                hyp = Hypothesis(
                    program=transform_obj,
                    name=f"spatial_{len(hypotheses)}",
                    parameters={'description': description, 'variant': 'spatial'},
                    activation=confidence
                )
                hypotheses.append(hyp)

        # === UNCONDITIONAL PATTERNS (from parent class) ===
        try:
            parent_hyps = super()._generate_hypotheses(train_pairs, test_input)
            hypotheses.extend(parent_hyps)
        except Exception as e:
            pass

        # === ENHANCED PATTERN INFERENCE ===
        # Try comprehensive analysis (conditional + looping + composite)
        try:
            enhanced_results = self.enhanced_inference.analyze_and_generate_hypotheses(train_pairs)

            # Add looping hypotheses
            for transform_obj, description, confidence in enhanced_results.get('looping', []):
                hyp = Hypothesis(
                    program=transform_obj,
                    name=f"looping_{len(hypotheses)}",
                    parameters={'description': description, 'variant': 'looping'},
                    activation=confidence
                )
                hypotheses.append(hyp)

            # Add composite hypotheses
            for transform_obj, description, confidence in enhanced_results.get('composite', []):
                hyp = Hypothesis(
                    program=transform_obj,
                    name=f"composite_{len(hypotheses)}",
                    parameters={'description': description, 'variant': 'composite'},
                    activation=confidence
                )
                hypotheses.append(hyp)

        except Exception as e:
            pass

        return hypotheses[:60]  # Increased limit to 60 for Phase 3

    def _generate_spatial_variations(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                    test_input: np.ndarray) -> List[Tuple[Transform, str, float]]:
        """
        Generate TRAINING-SPECIFIC spatial variations with validation.

        Key improvements:
        1. Extract colors from training OUTPUTS only (not all colors 0-9)
        2. Validate hypotheses on training data before adding
        3. Learn which spatial predicates matter for this task
        4. Generalize across tasks (not hardcoded)
        """
        variations = []

        # Detect objects in test input to understand spatial context
        detector = ObjectDetector()
        test_objects = detector.detect_objects(test_input)

        if not test_objects:
            return variations

        # IMPROVED: Extract colors from training OUTPUT grids only
        training_output_colors = set()
        for inp, out in train_pairs:
            training_output_colors.update(np.unique(out))

        # Remove background (0) and limit to most common colors
        training_output_colors.discard(0)

        # Sort by frequency in training outputs
        color_counts = Counter()
        for inp, out in train_pairs:
            for color in training_output_colors:
                color_counts[color] += np.sum(out == color)

        # Use top 5 most frequent colors from training outputs
        training_colors = [color for color, _ in color_counts.most_common(5)]

        if not training_colors:
            return variations

        # Helper function: Validate conditional on training data
        def validate_conditional(cond_transform, train_pairs):
            """Test conditional on training, return accuracy."""
            correct = 0
            for inp, out in train_pairs:
                try:
                    pred = cond_transform.apply(inp)
                    if np.array_equal(pred, out):
                        correct += 1
                    elif pred.shape == out.shape:
                        # Partial credit
                        correct += (pred == out).mean()
                except:
                    pass
            return correct / len(train_pairs) if train_pairs else 0.0

        # Generate and VALIDATE spatial conditional variations
        # 1. Edge-based conditionals (VALIDATED)
        for color in training_colors:
            # IF near edge THEN recolor to color
            cond_transform = ConditionalTransform(
                condition=self.condition_lib.near_edge(2),
                then_action=self.action_lib.recolor_to(color),
                else_action=self.action_lib.keep(),
                confidence=0.6
            )

            # VALIDATE on training data
            accuracy = validate_conditional(cond_transform, train_pairs)

            # Only add if achieves reasonable accuracy on training
            if accuracy >= 0.3:  # At least 30% match on training
                def make_transform(ct=cond_transform):
                    def transform_fn(grid: np.ndarray) -> np.ndarray:
                        return ct.apply(grid, objects=None)
                    return transform_fn

                transform_obj = Transform(
                    name=f"spatial_edge_recolor_{color}",
                    function=make_transform(),
                    parameters={'variant': 'spatial', 'condition': 'near_edge'},
                    category='spatial_conditional'
                )

                # Use validated accuracy as confidence
                variations.append((transform_obj, f"IF near edge THEN recolor to {color}", accuracy))

        # 2. Size-based conditionals (VALIDATED)
        if len(test_objects) > 1 and len(training_colors) >= 2:
            sizes = [obj.size for obj in test_objects]
            median_size = int(np.median(sizes))

            # Try size-based color conditionals with top 2 colors
            for i, color1 in enumerate(training_colors[:2]):
                for color2 in training_colors[i+1:i+2]:  # Avoid duplicate pairs
                    # IF size > median THEN recolor to color1 ELSE color2
                    cond_transform = ConditionalTransform(
                        condition=self.condition_lib.size_greater_than(median_size),
                        then_action=self.action_lib.recolor_to(color1),
                        else_action=self.action_lib.recolor_to(color2),
                        confidence=0.6
                    )

                    # VALIDATE on training data
                    accuracy = validate_conditional(cond_transform, train_pairs)

                    # Only add if achieves reasonable accuracy
                    if accuracy >= 0.3:
                        def make_transform(ct=cond_transform):
                            def transform_fn(grid: np.ndarray) -> np.ndarray:
                                return ct.apply(grid, objects=None)
                            return transform_fn

                        transform_obj = Transform(
                            name=f"spatial_size_recolor_{color1}_{color2}",
                            function=make_transform(),
                            parameters={'variant': 'spatial', 'condition': 'size_gt'},
                            category='spatial_conditional'
                        )

                        variations.append((
                            transform_obj,
                            f"IF size > {median_size} THEN recolor to {color1} ELSE recolor to {color2}",
                            accuracy
                        ))

        # Return only VALIDATED variations (sorted by accuracy)
        variations.sort(key=lambda x: x[2], reverse=True)  # Sort by confidence
        return variations[:15]  # Limit to top 15 validated spatial variations

    def _generate_nested_conditionals(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                     test_input: np.ndarray) -> List[Tuple[Transform, str, float]]:
        """
        Generate NESTED conditional hypotheses (AND/OR/NOT logic).

        Phase 3 Priority 1: Expected +2-4% accuracy improvement.
        """
        nested_hyps = []

        # Extract training colors
        training_colors = self._extract_training_colors(train_pairs)
        if len(training_colors) < 2:
            return nested_hyps

        # Detect test objects
        detector = ObjectDetector()
        test_objects = detector.detect_objects(test_input)
        if len(test_objects) < 2:
            return nested_hyps

        # Get median size for conditionals
        sizes = [obj.size for obj in test_objects]
        median_size = int(np.median(sizes))

        # Helper: Validate on training
        def validate(cond_transform):
            correct = 0
            for inp, out in train_pairs:
                try:
                    pred = cond_transform.apply(inp)
                    if np.array_equal(pred, out):
                        correct += 1
                    elif pred.shape == out.shape:
                        correct += (pred == out).mean()
                except:
                    pass
            return correct / len(train_pairs) if train_pairs else 0.0

        # Generate AND conditionals: IF (A AND B) THEN C
        # Example: IF (large AND near_edge) THEN recolor to X
        for color in training_colors[:2]:
            # Create: IF (size > median AND near_edge) THEN recolor
            and_condition = create_and_condition(
                self.condition_lib.size_greater_than(median_size),
                self.condition_lib.near_edge(2)
            )

            cond_transform = ConditionalTransform(
                condition=and_condition,
                then_action=self.action_lib.recolor_to(color),
                else_action=self.action_lib.keep(),
                confidence=0.7
            )

            # Validate
            accuracy = validate(cond_transform)
            if accuracy >= 0.3:
                def make_transform(ct=cond_transform):
                    def transform_fn(grid: np.ndarray) -> np.ndarray:
                        return ct.apply(grid, objects=None)
                    return transform_fn

                transform_obj = Transform(
                    name=f"nested_and_size_edge_{color}",
                    function=make_transform(),
                    parameters={'variant': 'nested', 'logic': 'AND'},
                    category='nested_conditional'
                )

                nested_hyps.append((
                    transform_obj,
                    f"IF (size > {median_size} AND near edge) THEN recolor to {color}",
                    accuracy
                ))

        # Generate OR conditionals: IF (A OR B) THEN C
        # Example: IF (blue OR red) THEN move_to_center
        if len(training_colors) >= 2:
            # Create: IF (color1 OR color2) THEN move_to_center
            or_condition = create_or_condition(
                self.condition_lib.has_color(training_colors[0]),
                self.condition_lib.has_color(training_colors[1])
            )

            cond_transform = ConditionalTransform(
                condition=or_condition,
                then_action=self.action_lib.move_to_center(),
                else_action=self.action_lib.keep(),
                confidence=0.6
            )

            accuracy = validate(cond_transform)
            if accuracy >= 0.3:
                def make_transform(ct=cond_transform):
                    def transform_fn(grid: np.ndarray) -> np.ndarray:
                        return ct.apply(grid, objects=None)
                    return transform_fn

                transform_obj = Transform(
                    name=f"nested_or_colors_{training_colors[0]}_{training_colors[1]}",
                    function=make_transform(),
                    parameters={'variant': 'nested', 'logic': 'OR'},
                    category='nested_conditional'
                )

                nested_hyps.append((
                    transform_obj,
                    f"IF (color {training_colors[0]} OR color {training_colors[1]}) THEN move to center",
                    accuracy
                ))

        # Generate NOT conditionals: IF NOT A THEN B
        # Example: IF NOT near_edge THEN recolor (i.e., center objects)
        for color in training_colors[:2]:
            not_condition = create_not_condition(
                self.condition_lib.near_edge(2)
            )

            cond_transform = ConditionalTransform(
                condition=not_condition,
                then_action=self.action_lib.recolor_to(color),
                else_action=self.action_lib.keep(),
                confidence=0.6
            )

            accuracy = validate(cond_transform)
            if accuracy >= 0.3:
                def make_transform(ct=cond_transform):
                    def transform_fn(grid: np.ndarray) -> np.ndarray:
                        return ct.apply(grid, objects=None)
                    return transform_fn

                transform_obj = Transform(
                    name=f"nested_not_edge_{color}",
                    function=make_transform(),
                    parameters={'variant': 'nested', 'logic': 'NOT'},
                    category='nested_conditional'
                )

                nested_hyps.append((
                    transform_obj,
                    f"IF NOT near edge THEN recolor to {color}",
                    accuracy
                ))

        # Sort by accuracy
        nested_hyps.sort(key=lambda x: x[2], reverse=True)
        return nested_hyps[:10]  # Top 10 nested conditionals

    def _generate_multi_stage_pipelines(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                       test_input: np.ndarray) -> List[Tuple[Transform, str, float]]:
        """
        Generate MULTI-STAGE pipeline hypotheses.

        Phase 3 Priority 2: Expected +2-3% accuracy improvement.
        """
        pipeline_hyps = []

        # Extract training colors
        training_colors = self._extract_training_colors(train_pairs)
        if len(training_colors) < 2:
            return pipeline_hyps

        # Detect test objects
        detector = ObjectDetector()
        test_objects = detector.detect_objects(test_input)
        if not test_objects:
            return pipeline_hyps

        # Helper: Validate pipeline
        def validate_pipeline(pipeline):
            correct = 0
            for inp, out in train_pairs:
                try:
                    pred = pipeline.apply(inp)
                    if np.array_equal(pred, out):
                        correct += 1
                    elif pred.shape == out.shape:
                        correct += (pred == out).mean()
                except:
                    pass
            return correct / len(train_pairs) if train_pairs else 0.0

        # Generate 2-stage pipelines
        # Pattern 1: Color edges, then move colored objects
        if len(training_colors) >= 1:
            pipeline = ConditionalPipeline(name="edge_color_then_move")

            # Stage 1: IF near_edge THEN recolor to training_colors[0]
            stage1 = ConditionalStage(
                conditional=ConditionalTransform(
                    condition=self.condition_lib.near_edge(2),
                    then_action=self.action_lib.recolor_to(training_colors[0]),
                    else_action=self.action_lib.keep()
                ),
                name="edge_coloring",
                description=f"Color edges to {training_colors[0]}"
            )
            pipeline.add_stage(stage1)

            # Stage 2: IF has_color(training_colors[0]) THEN move_to_center
            stage2 = ConditionalStage(
                conditional=ConditionalTransform(
                    condition=self.condition_lib.has_color(training_colors[0]),
                    then_action=self.action_lib.move_to_center(),
                    else_action=self.action_lib.keep()
                ),
                name="move_colored",
                description=f"Move color {training_colors[0]} to center"
            )
            pipeline.add_stage(stage2)

            # Validate
            accuracy = validate_pipeline(pipeline)
            if accuracy >= 0.3:
                def make_transform(p=pipeline):
                    def transform_fn(grid: np.ndarray) -> np.ndarray:
                        return p.apply(grid)
                    return transform_fn

                transform_obj = Transform(
                    name=f"pipeline_edge_color_move_{training_colors[0]}",
                    function=make_transform(),
                    parameters={'variant': 'pipeline', 'stages': 2},
                    category='multi_stage'
                )

                pipeline_hyps.append((
                    transform_obj,
                    f"Stage1: Color edges {training_colors[0]} → Stage2: Move to center",
                    accuracy
                ))

        # Pattern 2: Size-based coloring, then position-based transformation
        if len(training_colors) >= 2:
            sizes = [obj.size for obj in test_objects]
            median_size = int(np.median(sizes))

            pipeline = ConditionalPipeline(name="size_color_then_position")

            # Stage 1: IF size > median THEN color1 ELSE color2
            stage1 = ConditionalStage(
                conditional=ConditionalTransform(
                    condition=self.condition_lib.size_greater_than(median_size),
                    then_action=self.action_lib.recolor_to(training_colors[0]),
                    else_action=self.action_lib.recolor_to(training_colors[1])
                ),
                name="size_based_coloring",
                description=f"Large→{training_colors[0]}, Small→{training_colors[1]}"
            )
            pipeline.add_stage(stage1)

            # Stage 2: Move specific color
            stage2 = ConditionalStage(
                conditional=ConditionalTransform(
                    condition=self.condition_lib.has_color(training_colors[0]),
                    then_action=self.action_lib.move_by(1, 0),  # Move right
                    else_action=self.action_lib.keep()
                ),
                name="move_large",
                description=f"Move {training_colors[0]} right"
            )
            pipeline.add_stage(stage2)

            # Validate
            accuracy = validate_pipeline(pipeline)
            if accuracy >= 0.3:
                def make_transform(p=pipeline):
                    def transform_fn(grid: np.ndarray) -> np.ndarray:
                        return p.apply(grid)
                    return transform_fn

                transform_obj = Transform(
                    name=f"pipeline_size_color_move",
                    function=make_transform(),
                    parameters={'variant': 'pipeline', 'stages': 2},
                    category='multi_stage'
                )

                pipeline_hyps.append((
                    transform_obj,
                    f"Stage1: Size→Color → Stage2: Move colored",
                    accuracy
                ))

        # Sort by accuracy
        pipeline_hyps.sort(key=lambda x: x[2], reverse=True)
        return pipeline_hyps[:5]  # Top 5 pipelines

    def _extract_training_colors(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[int]:
        """Helper: Extract colors from training outputs."""
        training_output_colors = set()
        for inp, out in train_pairs:
            training_output_colors.update(np.unique(out))

        training_output_colors.discard(0)

        color_counts = Counter()
        for inp, out in train_pairs:
            for color in training_output_colors:
                color_counts[color] += np.sum(out == color)

        return [color for color, _ in color_counts.most_common(5)]




def test_conditional_solver():
    """Test the conditional solver on sample tasks."""
    solver = ConditionalARCCuriositySolver()

    # Load a sample task
    data_dir = Path("ARC-AGI/data/training")
    if not data_dir.exists():
        print("ARC-AGI data not found")
        return

    import json

    # Test on a task
    task_files = list(data_dir.glob("*.json"))[:5]

    print("Testing Conditional Solver\n")
    print("=" * 60)

    for task_file in task_files:
        with open(task_file, 'r') as f:
            task_data = json.load(f)

        task_id = task_file.stem

        try:
            predictions = solver.solve_task(task_data, verbose=False)
            expected = np.array(task_data['test'][0]['output'])

            # Check accuracy
            match1 = np.array_equal(predictions[0], expected)
            match2 = np.array_equal(predictions[1], expected)

            if match1 or match2:
                accuracy = 100.0
                symbol = "✓"
            else:
                accuracy = (predictions[0] == expected).mean() * 100
                symbol = " "

            print(f"{symbol} {task_id}: {accuracy:.1f}% accurate")

        except Exception as e:
            print(f"✗ {task_id}: ERROR - {e}")

    print("=" * 60)


if __name__ == '__main__':
    test_conditional_solver()
