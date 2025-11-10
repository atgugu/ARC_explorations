"""
Multi-Stage Conditional Compositions

Enables sequential conditional transformations:
- Stage 1: IF near_edge THEN recolor to blue
- Stage 2: IF blue AND large THEN move to center
- Stage 3: IF at_center THEN scale 2x

This is Phase 3 Priority 2 - expected +2-3% accuracy improvement.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from ..transformations.conditional_transforms import ConditionalTransform
from ..core.object_reasoning import ObjectDetector


@dataclass
class ConditionalStage:
    """A single stage in a multi-stage transformation."""
    conditional: ConditionalTransform
    name: str
    description: str

    def apply(self, grid: np.ndarray, objects: Optional[List] = None) -> np.ndarray:
        """Apply this stage's conditional transformation."""
        return self.conditional.apply(grid, objects)


class ConditionalPipeline:
    """
    Sequential pipeline of conditional transformations.

    Each stage can depend on the output of previous stages.

    Example:
        pipeline = ConditionalPipeline()
        pipeline.add_stage(stage1)  # IF near_edge THEN blue
        pipeline.add_stage(stage2)  # IF blue THEN move_center
        result = pipeline.apply(input_grid)
    """

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self.stages: List[ConditionalStage] = []

    def add_stage(self, stage: ConditionalStage):
        """Add a transformation stage to the pipeline."""
        self.stages.append(stage)

    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply all stages in sequence."""
        result = grid.copy()

        for i, stage in enumerate(self.stages):
            try:
                # Detect objects at this stage (may have changed)
                detector = ObjectDetector()
                objects = detector.detect_objects(result)

                # Apply stage
                result = stage.apply(result, objects)

            except Exception as e:
                # If stage fails, continue with current result
                continue

        return result

    def __repr__(self):
        stage_names = [s.name for s in self.stages]
        return f"ConditionalPipeline({' → '.join(stage_names)})"


class PipelineBuilder:
    """
    Builder for inferring multi-stage pipelines from training data.

    Strategy: Try to decompose complex transformations into stages.
    """

    def __init__(self):
        self.object_detector = ObjectDetector()

    def detect_multi_stage_patterns(self, train_pairs: List[tuple]) -> List[ConditionalPipeline]:
        """
        Detect if training examples show multi-stage transformations.

        Approach:
        1. Try 2-stage decomposition: intermediate → output
        2. Check if each stage has clear conditional pattern
        3. Validate on training data
        """
        pipelines = []

        # For now, we'll generate some common pipeline templates
        # More sophisticated detection can be added later

        # Common pipeline patterns:
        # 1. Color then move
        # 2. Filter then transform
        # 3. Transform then combine

        return pipelines

    def validate_pipeline(self, pipeline: ConditionalPipeline,
                         train_pairs: List[tuple]) -> float:
        """
        Validate pipeline on training data.

        Returns: Accuracy (0.0 to 1.0)
        """
        correct = 0
        total = len(train_pairs)

        for inp, out in train_pairs:
            try:
                pred = pipeline.apply(inp)

                if np.array_equal(pred, out):
                    correct += 1
                elif pred.shape == out.shape:
                    accuracy = (pred == out).mean()
                    correct += accuracy
            except:
                pass

        return correct / total if total > 0 else 0.0


class ConditionalLoop:
    """
    FOR-each loop applying conditional to multiple objects/regions.

    Example:
        loop = ConditionalLoop(
            iterator='objects',
            conditional=IF near_edge THEN recolor_blue
        )
        result = loop.apply(grid)
    """

    def __init__(self, iterator_type: str, conditional: ConditionalTransform):
        """
        Args:
            iterator_type: 'objects', 'regions_3x3', 'rows', 'columns'
            conditional: Conditional to apply to each element
        """
        self.iterator_type = iterator_type
        self.conditional = conditional

    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply conditional to each element."""
        result = grid.copy()

        if self.iterator_type == 'objects':
            # FOR each object: apply conditional
            detector = ObjectDetector()
            objects = detector.detect_objects(grid)

            for obj in objects:
                result = self.conditional.apply(result, objects=[obj])

        elif self.iterator_type == 'regions_3x3':
            # FOR each 3x3 region: apply conditional
            h, w = grid.shape
            for i in range(0, h, 3):
                for j in range(0, w, 3):
                    region = result[i:min(i+3, h), j:min(j+3, w)]
                    transformed = self.conditional.apply(region)
                    result[i:min(i+3, h), j:min(j+3, w)] = transformed

        elif self.iterator_type == 'rows':
            # FOR each row: apply conditional
            for i in range(grid.shape[0]):
                row = result[i:i+1, :].copy()
                transformed = self.conditional.apply(row)
                result[i:i+1, :] = transformed

        elif self.iterator_type == 'columns':
            # FOR each column: apply conditional
            for j in range(grid.shape[1]):
                col = result[:, j:j+1].copy()
                transformed = self.conditional.apply(col)
                result[:, j:j+1] = transformed

        return result

    def __repr__(self):
        return f"ConditionalLoop({self.iterator_type}, {self.conditional.description})"


# Example usage:
if __name__ == '__main__':
    from ..transformations.conditional_transforms import (
        ConditionalTransform, ConditionLibrary, ActionLibrary
    )

    lib_cond = ConditionLibrary()
    lib_action = ActionLibrary()

    # Create a 2-stage pipeline
    pipeline = ConditionalPipeline(name="color_then_move")

    # Stage 1: IF near_edge THEN recolor to blue
    stage1 = ConditionalStage(
        conditional=ConditionalTransform(
            condition=lib_cond.near_edge(2),
            then_action=lib_action.recolor_to(1),  # Blue
            else_action=lib_action.keep()
        ),
        name="edge_coloring",
        description="Color edges blue"
    )
    pipeline.add_stage(stage1)

    # Stage 2: IF blue AND large THEN move to center
    from ..transformations.nested_conditionals import create_and_condition

    stage2 = ConditionalStage(
        conditional=ConditionalTransform(
            condition=create_and_condition(
                lib_cond.has_color(1),  # Blue
                lib_cond.size_greater_than(5)  # Large
            ),
            then_action=lib_action.move_to_center(),
            else_action=lib_action.keep()
        ),
        name="move_blue_large",
        description="Move large blue objects to center"
    )
    pipeline.add_stage(stage2)

    print(f"Created pipeline: {pipeline}")
    print(f"  Stage 1: {stage1.description}")
    print(f"  Stage 2: {stage2.description}")
