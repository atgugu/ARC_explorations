"""Diagnose action detection to understand why Phase 6 performs worse."""

import json
import numpy as np
from pathlib import Path

from arc_curiosity_solver.core.action_inference import ActionInference


def main():
    print("\n" + "="*80)
    print("ACTION DETECTION DIAGNOSTIC")
    print("="*80)
    print("\nAnalyzing what actions are detected in training data...\n")

    training_dir = Path("ARC-AGI/data/training")
    task_files = sorted(list(training_dir.glob("*.json")))[:30]

    action_inference = ActionInference()

    detection_stats = {
        'rotations': 0,
        'reflections': 0,
        'color_swaps': 0,
        'extensions': 0,
        'replications': 0,
        'size_changes': 0,
        'position_changes': 0,
        'removals': 0
    }

    for i, task_file in enumerate(task_files):
        task_id = task_file.stem

        with open(task_file, 'r') as f:
            task_data = json.load(f)

        train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                      for ex in task_data['train']]

        try:
            detected = action_inference.analyze_training_pairs(train_pairs)

            # Count what was detected
            has_any = False
            detected_list = []

            if detected.get('rotations'):
                detection_stats['rotations'] += 1
                has_any = True
                detected_list.append(f"rotations: {list(detected['rotations'])}")

            if detected.get('reflections'):
                detection_stats['reflections'] += 1
                has_any = True
                detected_list.append(f"reflections: {list(detected['reflections'])}")

            if detected.get('color_swaps'):
                detection_stats['color_swaps'] += 1
                has_any = True
                detected_list.append(f"color_swaps: {len(detected['color_swaps'])} mappings")

            if detected.get('extensions'):
                detection_stats['extensions'] += 1
                has_any = True
                detected_list.append(f"extensions: {list(detected['extensions'])}")

            if detected.get('replications'):
                detection_stats['replications'] += 1
                has_any = True
                detected_list.append("replications")

            if detected.get('size_changes'):
                detection_stats['size_changes'] += 1
                has_any = True
                detected_list.append("size_changes")

            if detected.get('position_changes'):
                detection_stats['position_changes'] += 1
                has_any = True
                detected_list.append(f"position: {list(detected['position_changes'])}")

            if detected.get('removals'):
                detection_stats['removals'] += 1
                has_any = True
                detected_list.append("removals")

            if has_any:
                print(f"✓ {task_id}: {', '.join(detected_list)}")
            else:
                print(f"✗ {task_id}: NO ACTIONS DETECTED")

        except Exception as e:
            print(f"✗ {task_id}: ERROR - {e}")

    print("\n" + "="*80)
    print("DETECTION STATISTICS")
    print("="*80)

    total_tasks = len(task_files)

    print(f"\nAction detection rates across {total_tasks} tasks:")
    print(f"  Rotations:        {detection_stats['rotations']:2d}/{total_tasks} ({detection_stats['rotations']/total_tasks*100:.1f}%)")
    print(f"  Reflections:      {detection_stats['reflections']:2d}/{total_tasks} ({detection_stats['reflections']/total_tasks*100:.1f}%)")
    print(f"  Color swaps:      {detection_stats['color_swaps']:2d}/{total_tasks} ({detection_stats['color_swaps']/total_tasks*100:.1f}%)")
    print(f"  Extensions:       {detection_stats['extensions']:2d}/{total_tasks} ({detection_stats['extensions']/total_tasks*100:.1f}%)")
    print(f"  Replications:     {detection_stats['replications']:2d}/{total_tasks} ({detection_stats['replications']/total_tasks*100:.1f}%)")
    print(f"  Size changes:     {detection_stats['size_changes']:2d}/{total_tasks} ({detection_stats['size_changes']/total_tasks*100:.1f}%)")
    print(f"  Position changes: {detection_stats['position_changes']:2d}/{total_tasks} ({detection_stats['position_changes']/total_tasks*100:.1f}%)")
    print(f"  Removals:         {detection_stats['removals']:2d}/{total_tasks} ({detection_stats['removals']/total_tasks*100:.1f}%)")

    # Tasks with no detected composite actions
    tasks_with_composite = (detection_stats['rotations'] + detection_stats['reflections'] +
                           detection_stats['color_swaps'] + detection_stats['extensions'] +
                           detection_stats['replications'])

    # Note: Some tasks may have multiple types detected
    print(f"\n⚠️  CONCERN: If composite actions are rarely detected, we're skipping")
    print(f"    hypotheses that could help. Detection might be too conservative.")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
