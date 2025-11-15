"""
Debug script to check if extend_markers is being generated and scored
"""

import json
import numpy as np
from pathlib import Path
from parameter_inference import ParameterInference
from extend_markers_primitive import infer_extension_parameters, extend_markers
from inferred_solver import InferredProgramGenerator
from advanced_solver import AdvancedProgramGenerator

def load_task(task_id: str):
    """Load task from arc_data"""
    task_path = Path("arc_data/evaluation") / f"{task_id}.json"
    with open(task_path, 'r') as f:
        data = json.load(f)
    return {'train': data['train'], 'test': data['test']}

def debug_task(task_id: str):
    """Debug candidate generation for a task"""
    print(f"\n{'='*70}")
    print(f"Debugging task: {task_id}")
    print(f"{'='*70}")

    task = load_task(task_id)

    # 1. Check parameter inference
    print("\n1. PARAMETER INFERENCE")
    print("-" * 70)

    inferred = ParameterInference.infer_all_parameters(task)
    print(f"Inferred parameters: {inferred}")

    if inferred.extension_params:
        print(f"\nExtension parameters:")
        for key, value in inferred.extension_params.items():
            print(f"  {key}: {value}")
    else:
        print("\n❌ WARNING: No extension parameters inferred!")

    # 2. Test extend_markers directly
    print("\n2. TESTING EXTEND_MARKERS DIRECTLY")
    print("-" * 70)

    if inferred.extension_params:
        # Apply to first training example
        input_grid = np.array(task['train'][0]['input'])
        output_grid = np.array(task['train'][0]['output'])

        print(f"Input shape: {input_grid.shape}")
        print(f"Output shape: {output_grid.shape}")

        # Apply extend_markers
        result = extend_markers(
            input_grid,
            marker_colors=inferred.extension_params.get('marker_colors'),
            base_color=inferred.extension_params.get('base_color'),
            directions=inferred.extension_params.get('directions'),
            distance=inferred.extension_params.get('distance', 1)
        )

        # Calculate accuracy
        accuracy = np.sum(result == output_grid) / output_grid.size

        print(f"\nResult accuracy on training example 1: {accuracy:.2%}")

        if accuracy > 0.95:
            print("✓ extend_markers works well on training!")
        else:
            print("❌ extend_markers doesn't match training output")

    # 3. Check candidate generation
    print("\n3. CANDIDATE GENERATION")
    print("-" * 70)

    generator = InferredProgramGenerator()
    candidates = generator.generate_candidates(task, max_candidates=200)

    # Find extend_markers candidates
    extend_markers_candidates = [c for c in candidates if c.schema == "extend_markers" or (hasattr(c, 'steps') and any(s.schema == "extend_markers" for s in c.steps))]

    print(f"Total candidates generated: {len(candidates)}")
    print(f"extend_markers candidates: {len(extend_markers_candidates)}")

    if extend_markers_candidates:
        print(f"\nextend_markers candidates found:")
        for i, cand in enumerate(extend_markers_candidates[:5], 1):
            print(f"  {i}. Schema: {cand.schema}")
            print(f"     Parameters: {cand.parameters}")
            print(f"     Complexity: {cand.complexity}")
    else:
        print("\n❌ WARNING: No extend_markers candidates generated!")

    # Check compositional candidates with extend_markers
    compositional_with_extend = [c for c in candidates if hasattr(c, 'steps') and c.steps and any(s.schema == "extend_markers" for s in c.steps)]

    if compositional_with_extend:
        print(f"\nCompositional candidates with extend_markers: {len(compositional_with_extend)}")
        for i, cand in enumerate(compositional_with_extend[:3], 1):
            steps_str = " → ".join([s.schema for s in cand.steps])
            print(f"  {i}. {steps_str}")

def main():
    """Debug the 3 near-miss tasks"""
    tasks = ["fd096ab6", "42918530", "e681b708"]

    for task_id in tasks:
        debug_task(task_id)

if __name__ == "__main__":
    main()
