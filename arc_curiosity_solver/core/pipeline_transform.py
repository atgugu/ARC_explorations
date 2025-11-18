"""Multi-Stage Pipeline Transformations

Chains multiple transformations in sequence to handle tasks requiring
sequential reasoning. This is Phase 7.

Example pipeline:
  Input → Stage 1 (rotate) → Stage 2 (recolor) → Stage 3 (extend) → Output
"""

import numpy as np
from typing import List, Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PipelineStage:
    """A single stage in a transformation pipeline."""
    transform: Callable[[np.ndarray], np.ndarray]
    name: str
    description: str
    confidence: float = 0.5


class PipelineTransform:
    """
    Multi-stage transformation pipeline.

    Applies transformations sequentially:
    output = stage_N(...stage_2(stage_1(input)))
    """

    def __init__(self, stages: List[PipelineStage], name: str = "pipeline"):
        """
        Initialize pipeline with ordered stages.

        Args:
            stages: List of PipelineStage objects to apply in order
            name: Name for this pipeline
        """
        self.stages = stages
        self.name = name

        # Calculate overall confidence (product of stage confidences)
        self.confidence = 1.0
        for stage in stages:
            self.confidence *= stage.confidence

    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply all stages sequentially."""
        result = grid.copy()

        for stage in self.stages:
            try:
                result = stage.transform(result)
            except Exception as e:
                # If any stage fails, return what we have so far
                return result

        return result

    def apply_partial(self, grid: np.ndarray, num_stages: int) -> np.ndarray:
        """Apply only the first num_stages."""
        result = grid.copy()

        for stage in self.stages[:num_stages]:
            try:
                result = stage.transform(result)
            except Exception as e:
                return result

        return result

    def get_description(self) -> str:
        """Get human-readable description of pipeline."""
        stage_descs = [f"Stage {i+1}: {s.description}"
                      for i, s in enumerate(self.stages)]
        return " → ".join(stage_descs)

    def __len__(self) -> int:
        """Number of stages in pipeline."""
        return len(self.stages)


class PipelineGenerator:
    """
    Generates multi-stage pipelines from single-stage transformations.

    Uses greedy search to build pipelines:
    1. Find best stage 1 (gets closest to output)
    2. Find best stage 2 given stage 1
    3. Continue for N stages
    """

    def __init__(self, max_stages: int = 3, beam_width: int = 5):
        """
        Initialize pipeline generator.

        Args:
            max_stages: Maximum pipeline length (2-3 recommended)
            beam_width: Number of candidates to keep at each stage
        """
        self.max_stages = max_stages
        self.beam_width = beam_width

    def generate_2stage_pipelines(
        self,
        stage1_transforms: List[Tuple],  # (transform, name, confidence)
        stage2_transforms: List[Tuple],
        train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[PipelineTransform, float]]:
        """
        Generate 2-stage pipelines using greedy search.

        Args:
            stage1_transforms: Candidate first-stage transformations
            stage2_transforms: Candidate second-stage transformations
            train_pairs: Training examples for validation

        Returns:
            List of (pipeline, accuracy) tuples
        """
        pipelines = []

        # Evaluate all stage 1 candidates
        stage1_scores = []
        for transform, name, conf in stage1_transforms[:self.beam_width * 2]:
            score = self._evaluate_partial(transform, train_pairs, target_stage=1)
            stage1_scores.append((transform, name, conf, score))

        # Keep top beam_width stage 1 candidates
        stage1_scores.sort(key=lambda x: x[3], reverse=True)
        top_stage1 = stage1_scores[:self.beam_width]

        # For each good stage 1, try stage 2 options
        for s1_transform, s1_name, s1_conf, s1_score in top_stage1:
            # Skip if stage 1 doesn't improve anything
            if s1_score < 0.1:
                continue

            # Try stage 2 candidates
            for s2_transform, s2_name, s2_conf in stage2_transforms[:20]:
                # Create 2-stage pipeline
                pipeline = PipelineTransform(
                    stages=[
                        PipelineStage(s1_transform, s1_name, s1_name, s1_conf),
                        PipelineStage(s2_transform, s2_name, s2_name, s2_conf)
                    ],
                    name=f"{s1_name}_then_{s2_name}"
                )

                # Validate full pipeline
                accuracy = self._validate_pipeline(pipeline, train_pairs)

                if accuracy > 0.15:  # Threshold for keeping pipeline
                    pipelines.append((pipeline, accuracy))

        # Sort by accuracy
        pipelines.sort(key=lambda x: x[1], reverse=True)
        return pipelines[:20]  # Return top 20

    def generate_3stage_pipelines(
        self,
        transforms: List[Tuple],
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        top_2stage_pipelines: List[Tuple[PipelineTransform, float]]
    ) -> List[Tuple[PipelineTransform, float]]:
        """
        Generate 3-stage pipelines by extending good 2-stage pipelines.

        Args:
            transforms: Candidate transformations for stage 3
            train_pairs: Training examples
            top_2stage_pipelines: Good 2-stage pipelines to extend

        Returns:
            List of (pipeline, accuracy) tuples
        """
        pipelines = []

        # Extend top 2-stage pipelines with a 3rd stage
        for pipeline_2stage, acc_2stage in top_2stage_pipelines[:5]:
            # Skip if 2-stage already good enough
            if acc_2stage > 0.8:
                continue

            # Try adding a 3rd stage
            for transform, name, conf in transforms[:15]:
                # Create 3-stage pipeline
                new_stages = pipeline_2stage.stages + [
                    PipelineStage(transform, name, name, conf)
                ]

                pipeline = PipelineTransform(
                    stages=new_stages,
                    name=f"{pipeline_2stage.name}_then_{name}"
                )

                # Validate
                accuracy = self._validate_pipeline(pipeline, train_pairs)

                if accuracy > acc_2stage:  # Must improve over 2-stage
                    pipelines.append((pipeline, accuracy))

        pipelines.sort(key=lambda x: x[1], reverse=True)
        return pipelines[:10]  # Return top 10

    def _evaluate_partial(
        self,
        transform: Callable,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        target_stage: int
    ) -> float:
        """
        Evaluate how well a partial transformation moves toward the output.

        Uses "improvement score": how much closer to output after transform.
        """
        total_improvement = 0.0

        for inp, out in train_pairs:
            try:
                # Apply transformation
                intermediate = transform(inp.copy())

                # Calculate improvement
                # Before: distance from input to output
                # After: distance from intermediate to output
                if inp.shape == out.shape:
                    before_dist = (inp != out).mean()
                else:
                    before_dist = 1.0

                if intermediate.shape == out.shape:
                    after_dist = (intermediate != out).mean()
                    improvement = max(0, before_dist - after_dist)
                    total_improvement += improvement
                else:
                    # Shape mismatch - but might be intentional
                    # Give small credit if not same as input
                    if intermediate.shape != inp.shape:
                        total_improvement += 0.1
            except:
                pass

        return total_improvement / len(train_pairs) if train_pairs else 0.0

    def _validate_pipeline(
        self,
        pipeline: PipelineTransform,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Validate complete pipeline on training pairs."""
        correct = 0.0

        for inp, out in train_pairs:
            try:
                pred = pipeline.apply(inp.copy())

                if np.array_equal(pred, out):
                    correct += 1.0
                elif pred.shape == out.shape:
                    correct += (pred == out).mean()
            except:
                pass

        return correct / len(train_pairs) if train_pairs else 0.0


def create_pipeline_from_transforms(
    transforms: List[Callable],
    names: List[str],
    confidences: Optional[List[float]] = None
) -> PipelineTransform:
    """
    Convenience function to create a pipeline from transformations.

    Args:
        transforms: List of transformation functions
        names: List of names for each stage
        confidences: Optional confidence scores (default 0.5)

    Returns:
        PipelineTransform object
    """
    if confidences is None:
        confidences = [0.5] * len(transforms)

    stages = [
        PipelineStage(t, n, n, c)
        for t, n, c in zip(transforms, names, confidences)
    ]

    return PipelineTransform(stages)
