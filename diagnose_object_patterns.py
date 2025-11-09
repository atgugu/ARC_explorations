"""
Diagnose what object-level patterns actually exist in failing ARC tasks.

Understand WHY the object transformations didn't help.
"""

import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/user/ARC_explorations')

from arc_curiosity_solver.core.object_reasoning import ObjectDetector, analyze_object_patterns


def visualize_objects(grid, objects, title=""):
    """Visualize detected objects."""
    print(f"\n{title}")
    print(f"Grid shape: {grid.shape}, Objects detected: {len(objects)}")

    for i, obj in enumerate(objects):
        print(f"\nObject {i+1}:")
        print(f"  Position: {obj.position}")
        print(f"  Size: {obj.size} pixels")
        print(f"  Colors: {obj.colors}")
        print(f"  Dominant color: {obj.dominant_color}")
        print(f"  Bbox: {obj.bbox}")
        print(f"  Shape: {obj.height}x{obj.width}, aspect={obj.aspect_ratio:.2f}, density={obj.density:.2f}")


def deep_pattern_analysis(task_file: str):
    """Deeply analyze what object patterns exist."""
    with open(task_file, 'r') as f:
        data = json.load(f)

    task_id = Path(task_file).stem
    print(f"\n{'='*70}")
    print(f"Deep Analysis: {task_id}")
    print(f"{'='*70}")

    detector = ObjectDetector()

    # Load training pairs
    train_pairs = []
    for example in data['train']:
        inp = np.array(example['input'])
        out = np.array(example['output'])
        train_pairs.append((inp, out))

    # Analyze patterns
    patterns = analyze_object_patterns(train_pairs)
    print(f"\nObject Pattern Detection:")
    print(f"  Needs object reasoning: {patterns['needs_object_reasoning']}")
    print(f"  Object count changes: {patterns['object_count_changes']}")
    print(f"  Has size changes: {len(patterns['size_changes']) > 0}")
    print(f"  Has color changes: {len(patterns['color_changes']) > 0}")
    print(f"  Has position changes: {len(patterns['position_changes']) > 0}")

    # Analyze each training pair in detail
    for i, (inp, out) in enumerate(train_pairs[:2]):  # First 2 examples
        print(f"\n--- Training Example {i+1} ---")

        inp_objects = detector.detect_objects(inp)
        out_objects = detector.detect_objects(out)

        visualize_objects(inp, inp_objects, f"Input Objects")
        visualize_objects(out, out_objects, f"Output Objects")

        # What changed?
        print(f"\nChanges:")
        print(f"  Object count: {len(inp_objects)} → {len(out_objects)}")

        if len(inp_objects) == len(out_objects) and len(inp_objects) > 0:
            # Match objects by position
            for j in range(min(len(inp_objects), len(out_objects))):
                inp_obj = inp_objects[j]
                out_obj = out_objects[j]

                if inp_obj.dominant_color != out_obj.dominant_color:
                    print(f"  Object {j+1} color changed: {inp_obj.dominant_color} → {out_obj.dominant_color}")

                if inp_obj.size != out_obj.size:
                    print(f"  Object {j+1} size changed: {inp_obj.size} → {out_obj.size}")

                pos_change = np.linalg.norm(np.array(inp_obj.position) - np.array(out_obj.position))
                if pos_change > 2:
                    print(f"  Object {j+1} position changed: {inp_obj.position} → {out_obj.position}")

        # What's the actual transformation?
        print(f"\nActual Transformation (pixel analysis):")

        # Check if it's a simple operation
        if inp.shape == out.shape:
            diff = np.sum(inp != out)
            print(f"  Same shape, {diff} pixels different")

            # Check for patterns
            if np.array_equal(inp, out):
                print(f"  → Identity transformation")
            elif diff < out.size * 0.1:
                print(f"  → Minor modification (<10% pixels changed)")
            else:
                print(f"  → Significant modification ({diff/out.size*100:.1f}% pixels changed)")

        else:
            print(f"  Shape change: {inp.shape} → {out.shape}")

            # Check if output contains input
            if out.shape[0] >= inp.shape[0] and out.shape[1] >= inp.shape[1]:
                # Check for scaling
                scale_y = out.shape[0] / inp.shape[0]
                scale_x = out.shape[1] / inp.shape[1]

                if scale_y == scale_x and scale_y.is_integer():
                    print(f"  → Likely {int(scale_y)}x scaling")

            elif out.shape[0] < inp.shape[0] or out.shape[1] < inp.shape[1]:
                print(f"  → Compression/extraction")


def main():
    """Diagnose key failing tasks."""

    training_dir = Path("/home/user/ARC_explorations/ARC-AGI/data/training")

    # Pick tasks that showed no improvement but have object patterns
    interesting_tasks = [
        "007bbfb7",  # 77.8% - has objects, no improvement
        "025d127b",  # 98% - very close, has objects
        "11852cab",  # 97% - very close, has objects
        "0b148d64",  # 0% - wrong shape, has objects
        "05269061",  # 22% - pattern task, has objects
    ]

    for task_id in interesting_tasks:
        task_file = training_dir / f"{task_id}.json"
        if task_file.exists():
            try:
                deep_pattern_analysis(str(task_file))
            except Exception as e:
                print(f"ERROR analyzing {task_id}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*70}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
