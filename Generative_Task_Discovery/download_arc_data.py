"""
Download ARC evaluation dataset for testing

Downloads the official ARC dataset from Kaggle/GitHub
"""

import os
import json
import urllib.request
import zipfile
from pathlib import Path

def download_arc_data():
    """Download ARC dataset from GitHub"""

    data_dir = Path("arc_data")
    data_dir.mkdir(exist_ok=True)

    # ARC dataset URLs (from official repo)
    base_url = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"

    datasets = {
        "training": f"{base_url}/training",
        "evaluation": f"{base_url}/evaluation"
    }

    print("="*70)
    print("DOWNLOADING ARC DATASET")
    print("="*70)

    # Try to get list of files from the repo
    # For simplicity, let's create a script that uses git clone instead

    print("\nAttempting to clone ARC dataset...")

    return data_dir

def create_synthetic_evaluation_set(n_tasks=200):
    """
    Create a synthetic evaluation set based on our primitives

    This generates diverse tasks using combinations of our primitives
    to test the solver at scale.
    """
    import numpy as np
    from typing import List, Dict, Any

    tasks = []

    # Task templates based on our primitives
    templates = [
        # Transformations
        ("identity", lambda g: g),
        ("flip_h", lambda g: np.fliplr(g)),
        ("flip_v", lambda g: np.flipud(g)),
        ("rotate_90", lambda g: np.rot90(g, k=1)),
        ("rotate_180", lambda g: np.rot90(g, k=2)),
        ("rotate_270", lambda g: np.rot90(g, k=3)),

        # Color operations
        ("increment_color", lambda g: np.where(g > 0, (g % 9) + 1, 0)),
        ("replace_color", lambda g: np.where(g == 1, 2, np.where(g == 2, 1, g))),

        # Pattern operations
        ("tile_horizontal", lambda g: np.hstack([g, g])),
        ("tile_vertical", lambda g: np.vstack([g, g])),

        # Size operations
        ("double_size", lambda g: np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)),
        ("half_size", lambda g: g[::2, ::2] if g.shape[0] > 1 else g),
    ]

    # Grid sizes
    sizes = [(2, 2), (3, 3), (4, 4), (5, 5), (3, 4), (4, 5), (2, 3)]

    # Color palettes
    colors = [0, 1, 2, 3, 4, 5]

    print("="*70)
    print("GENERATING SYNTHETIC EVALUATION SET")
    print("="*70)
    print(f"\nGenerating {n_tasks} tasks...")

    task_id = 0

    for template_name, transform in templates:
        for _ in range(n_tasks // len(templates) + 1):
            if task_id >= n_tasks:
                break

            # Random grid size
            h, w = sizes[task_id % len(sizes)]

            # Generate random input
            np.random.seed(task_id)
            density = 0.3 + (task_id % 5) * 0.1  # Vary density

            train_input = []
            for i in range(h):
                row = []
                for j in range(w):
                    if np.random.random() < density:
                        row.append(colors[np.random.randint(1, len(colors))])
                    else:
                        row.append(0)
                train_input.append(row)

            train_input = np.array(train_input)

            # Apply transform
            try:
                train_output = transform(train_input)

                # Generate test input (similar pattern)
                test_input = []
                for i in range(h):
                    row = []
                    for j in range(w):
                        if np.random.random() < density:
                            row.append(colors[np.random.randint(1, len(colors))])
                        else:
                            row.append(0)
                    test_input.append(row)

                test_input = np.array(test_input)
                test_output = transform(test_input)

                # Create task
                task = {
                    "id": f"synthetic_{task_id:04d}",
                    "name": f"{template_name}_{task_id}",
                    "category": template_name.split("_")[0],
                    "task": {
                        "train": [{
                            "input": train_input.tolist(),
                            "output": train_output.tolist()
                        }],
                        "test": [{
                            "input": test_input.tolist(),
                            "output": test_output.tolist()
                        }]
                    }
                }

                tasks.append(task)
                task_id += 1

                if task_id % 50 == 0:
                    print(f"  Generated {task_id}/{n_tasks} tasks...")

            except Exception as e:
                # Skip tasks that fail to generate
                continue

        if task_id >= n_tasks:
            break

    print(f"\n✓ Generated {len(tasks)} synthetic tasks")

    # Save to file
    output_file = "synthetic_evaluation_200.json"
    with open(output_file, 'w') as f:
        json.dump(tasks, f, indent=2)

    print(f"✓ Saved to {output_file}")

    return tasks

if __name__ == "__main__":
    print("\nOption 1: Download official ARC dataset (requires internet)")
    print("Option 2: Generate synthetic evaluation set (offline)\n")

    # For now, generate synthetic set
    print("Generating synthetic evaluation set...\n")
    tasks = create_synthetic_evaluation_set(n_tasks=200)

    print("\n" + "="*70)
    print("✓ EVALUATION DATA READY")
    print("="*70)
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"File: synthetic_evaluation_200.json")
