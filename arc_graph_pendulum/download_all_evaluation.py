"""
Download all ARC-AGI evaluation tasks.
"""

import requests
import json
from pathlib import Path
import time

# First 120 evaluation task IDs from the repository
EVALUATION_TASK_IDS = [
    "00576224", "009d5c81", "00dbd492", "03560426", "05a7bcf2", "0607ce86",
    "0692e18c", "070dd51e", "08573cc6", "0934a4d8", "09c534e7", "0a1d4ef5",
    "0a2355a6", "0b17323b", "0bb8deee", "0becf7df", "0c786b71", "0c9aba6e",
    "0d87d2a6", "0e671a1a", "0f63c0b9", "103eff5b", "11e1fe23", "12422b43",
    "12997ef3", "12eac192", "136b0064", "13713586", "137f0df0", "140c817e",
    "14754a24", "15113be4", "15663ba9", "15696249", "16b78196", "17b80ad2",
    "17cae0c1", "18419cfa", "184a9768", "195ba7dc", "1990f7a8", "19bb5feb",
    "1a2e2828", "1a6449f1", "1acc24af", "1c02dbbe", "1c0d0a4b", "1c56ad9f",
    "1d0a4b61", "1d398264", "1da012fc", "1e81d6f9", "1e97544e", "2037f2c7",
    "2072aba6", "20818e16", "20981f0e", "212895b5", "21f83797", "22a4bbc2",
    "25094a63", "2546ccf6", "256b0a75", "2685904e", "2697da3f", "2753e76c",
    "27a77e38", "27f8ce4f", "281123b4", "292dd178", "29700607", "2a5f8217",
    "2b01abd0", "2c0b0aff", "2c737e39", "2f0c5170", "310f3251", "3194b014",
    "319f2597", "31adaf00", "31d5ba1a", "32e9702f", "332efdb3", "3391f8c0",
    "33b52de3", "3490cc26", "34b99a2b", "351d6448", "358ba94e", "37d3e8b2",
    "3979b1a8", "3a301edc", "3b4c2228", "3d31c5b3", "3ed85e70", "3ee1011a",
    "3f23242b", "40f6cd08", "414297c0", "423a55dc", "42918530", "42a15761",
    "4364c1c4", "456873bc", "45737921", "45bbe264", "477d2879", "47996f11",
    "48131b3c", "4852f2fa", "48f8583b", "4aab4007", "4acc7107", "4b6b68e5",
    "4c177718", "4cd1b7b2", "4e45f183"
]

def download_all_evaluation():
    """Download evaluation tasks."""

    base_url = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/evaluation"
    local_dir = Path("./arc_data/evaluation")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(EVALUATION_TASK_IDS)} evaluation tasks...")
    print("="*60)

    downloaded = 0
    failed = 0
    skipped = 0

    for i, task_id in enumerate(EVALUATION_TASK_IDS, 1):
        task_url = f"{base_url}/{task_id}.json"
        file_path = local_dir / f"{task_id}.json"

        # Skip if already exists
        if file_path.exists():
            skipped += 1
            continue

        try:
            response = requests.get(task_url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

                downloaded += 1
                if downloaded % 10 == 0:
                    print(f"Progress: {downloaded}/{len(EVALUATION_TASK_IDS)} tasks downloaded...")
            else:
                print(f"✗ Failed {task_id} (status {response.status_code})")
                failed += 1

        except Exception as e:
            print(f"✗ Error {task_id}: {e}")
            failed += 1

        time.sleep(0.05)  # Be nice to GitHub

    print("="*60)
    print(f"Downloaded: {downloaded} new tasks")
    print(f"Skipped: {skipped} existing tasks")
    print(f"Failed: {failed} tasks")
    print(f"Total available: {downloaded + skipped} tasks")
    print("="*60)

    return downloaded + skipped > 0

if __name__ == "__main__":
    success = download_all_evaluation()

    if success:
        print("\n✓ Evaluation dataset ready!")
    else:
        print("\n✗ Failed to prepare evaluation dataset.")
