"""
Download the real ARC evaluation dataset from official source
"""

import os
import json
import urllib.request
import zipfile
from pathlib import Path
import shutil

def download_arc_dataset():
    """Download official ARC dataset from GitHub"""

    print("="*70)
    print("DOWNLOADING REAL ARC DATASET")
    print("="*70)

    # Create arc_data directory
    arc_dir = Path("arc_data")
    arc_dir.mkdir(exist_ok=True)

    # URLs for ARC data
    base_url = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"

    # Download evaluation tasks
    eval_dir = arc_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)

    print("\nAttempting to clone ARC repository...")
    print("This may take a moment...\n")

    # Use git clone to get the full dataset
    repo_url = "https://github.com/fchollet/ARC-AGI.git"
    temp_dir = "ARC-AGI-temp"

    try:
        # Clone the repository
        import subprocess
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print("✓ Repository cloned successfully")

            # Copy evaluation data
            src_eval = Path(temp_dir) / "data" / "evaluation"
            if src_eval.exists():
                # Copy all JSON files
                json_files = list(src_eval.glob("*.json"))
                print(f"\nCopying {len(json_files)} evaluation tasks...")

                for json_file in json_files:
                    shutil.copy(json_file, eval_dir / json_file.name)

                print(f"✓ Copied {len(json_files)} tasks to {eval_dir}")

                # Also copy training data (smaller sample)
                src_train = Path(temp_dir) / "data" / "training"
                train_dir = arc_dir / "training"
                train_dir.mkdir(exist_ok=True)

                if src_train.exists():
                    train_files = list(src_train.glob("*.json"))[:50]  # First 50 for quick testing
                    print(f"\nCopying {len(train_files)} training tasks...")

                    for json_file in train_files:
                        shutil.copy(json_file, train_dir / json_file.name)

                    print(f"✓ Copied {len(train_files)} training tasks to {train_dir}")

            # Clean up temp directory
            shutil.rmtree(temp_dir)
            print("\n✓ Cleanup complete")

            return True
        else:
            print(f"✗ Git clone failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("✗ Download timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def verify_dataset():
    """Verify the downloaded dataset"""

    print("\n" + "="*70)
    print("VERIFYING DATASET")
    print("="*70)

    arc_dir = Path("arc_data")
    eval_dir = arc_dir / "evaluation"
    train_dir = arc_dir / "training"

    if not eval_dir.exists():
        print("✗ Evaluation directory not found")
        return False

    eval_files = list(eval_dir.glob("*.json"))
    train_files = list(train_dir.glob("*.json")) if train_dir.exists() else []

    print(f"\nEvaluation tasks: {len(eval_files)}")
    print(f"Training tasks: {len(train_files)}")

    if len(eval_files) > 0:
        # Test loading one file
        test_file = eval_files[0]
        try:
            with open(test_file, 'r') as f:
                data = json.load(f)

            print(f"\n✓ Successfully loaded test file: {test_file.name}")
            print(f"  Train examples: {len(data.get('train', []))}")
            print(f"  Test examples: {len(data.get('test', []))}")

            # Show structure
            if data.get('train'):
                ex = data['train'][0]
                print(f"  Example input shape: {len(ex['input'])}x{len(ex['input'][0]) if ex['input'] else 0}")
                print(f"  Example output shape: {len(ex['output'])}x{len(ex['output'][0]) if ex['output'] else 0}")

            return True

        except Exception as e:
            print(f"✗ Error loading test file: {e}")
            return False
    else:
        print("✗ No evaluation files found")
        return False

if __name__ == "__main__":
    success = download_arc_dataset()

    if success:
        verify_dataset()

        print("\n" + "="*70)
        print("✓ DOWNLOAD COMPLETE")
        print("="*70)
        print("\nReady to evaluate on real ARC dataset!")
    else:
        print("\n" + "="*70)
        print("✗ DOWNLOAD FAILED")
        print("="*70)
        print("\nPlease check your internet connection and try again.")
