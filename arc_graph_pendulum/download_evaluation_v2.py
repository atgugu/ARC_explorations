"""
Download ARC-AGI evaluation dataset from GitHub.
"""

import requests
import json
from pathlib import Path
import time

def download_evaluation_tasks():
    """Download evaluation tasks from the ARC-AGI repository."""

    base_url = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/evaluation"
    local_dir = Path("./arc_data/evaluation")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Get list of evaluation tasks from the main data repository
    # First, try to get a list of known evaluation task IDs

    print("Attempting to download sample evaluation tasks...")

    # Try some known evaluation task IDs
    sample_task_ids = [
        "00576224", "009d5c81", "00d62c1b", "017c7c7b", "01e0c382",
        "0520fde7", "05269061", "05f2a901", "06df4c85", "08ed6ac7",
        "0962bcdd", "09629e4f", "0a938d79", "0b148d64", "0ca9ddb6",
        "0d3d703e", "0dfd9992", "0e206a2e", "10fcaaa3", "11852cab"
    ]

    downloaded = 0
    failed = 0

    for task_id in sample_task_ids:
        task_url = f"{base_url}/{task_id}.json"
        file_path = local_dir / f"{task_id}.json"

        # Skip if already exists
        if file_path.exists():
            print(f"✓ {task_id}.json already exists")
            downloaded += 1
            continue

        try:
            response = requests.get(task_url, timeout=10)

            if response.status_code == 200:
                # Validate JSON
                data = response.json()

                # Save to file
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

                print(f"✓ Downloaded {task_id}.json")
                downloaded += 1
            else:
                print(f"✗ Failed {task_id}.json (status {response.status_code})")
                failed += 1

        except Exception as e:
            print(f"✗ Error downloading {task_id}: {e}")
            failed += 1

        time.sleep(0.1)  # Be nice to GitHub

    print(f"\n{'='*50}")
    print(f"Downloaded: {downloaded} tasks")
    print(f"Failed: {failed} tasks")
    print(f"{'='*50}")

    return downloaded > 0

if __name__ == "__main__":
    success = download_evaluation_tasks()

    if success:
        print("\n✓ Successfully downloaded evaluation tasks!")
    else:
        print("\n✗ Failed to download evaluation tasks.")
