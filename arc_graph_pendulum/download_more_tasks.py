"""
Download a diverse set of ARC tasks for comprehensive testing.
"""

import requests
import json
from pathlib import Path
import time

# Broader set of task IDs from ARC training set
# Selected to cover diverse transformation types
DIVERSE_TASK_IDS = [
    # Original 10
    "0d3d703e", "1e0a9b12", "25ff71a9", "3c9b0459", "6150a2bd",
    "007bbfb7", "00d62c1b", "017c7c7b", "025d127b", "045e512c",

    # Additional diverse tasks (40 more for total of 50)
    "0520fde7", "05f2a901", "06df4c85", "08ed6ac7", "09629e4f",
    "0a938d79", "0b148d64", "0ca9ddb6", "0d87d2a6", "0e206a2e",
    "10fcaaa3", "11852cab", "1190e5a7", "136b0064", "137eaa0f",
    "150deff5", "178fcbfb", "1a2e2828", "1b2d62fb", "1b60fb0c",
    "1bfc4729", "1c786137", "1caeab9d", "1cf80156", "1e32b0e9",
    "1f0c79e5", "1f642eb9", "1f85a75f", "1f876c06", "1fad071e",
    "2013d3e2", "20818e16", "228f6490", "22233c11", "22eb0ac0",
    "234bbc79", "23581191", "239be575", "23b5c85d", "253bf280",
]

BASE_URL = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training"

def download_tasks(task_ids, cache_dir="./arc_data/training"):
    """Download tasks from GitHub."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failed = []

    for task_id in task_ids:
        file_path = cache_path / f"{task_id}.json"

        # Skip if already exists
        if file_path.exists():
            print(f"✓ {task_id} (cached)")
            downloaded += 1
            continue

        # Download
        url = f"{BASE_URL}/{task_id}.json"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(file_path, 'w') as f:
                    json.dump(response.json(), f, indent=2)
                print(f"✓ {task_id} (downloaded)")
                downloaded += 1
            else:
                print(f"✗ {task_id} (HTTP {response.status_code})")
                failed.append(task_id)
        except Exception as e:
            print(f"✗ {task_id} (error: {e})")
            failed.append(task_id)

        # Be nice to GitHub
        time.sleep(0.1)

    print(f"\n{'='*70}")
    print(f"Downloaded: {downloaded}/{len(task_ids)}")
    if failed:
        print(f"Failed: {len(failed)} tasks")
        print(f"  {', '.join(failed[:10])}")
    print(f"{'='*70}")

    return downloaded

if __name__ == "__main__":
    print("Downloading diverse ARC tasks for comprehensive testing...")
    print(f"Target: {len(DIVERSE_TASK_IDS)} tasks\n")

    downloaded = download_tasks(DIVERSE_TASK_IDS)

    print(f"\n✓ Ready to test on {downloaded} diverse tasks!")
