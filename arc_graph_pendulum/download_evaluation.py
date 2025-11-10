"""
Download the ARC-AGI evaluation dataset.
"""

from utils.arc_loader import ARCLoader

def main():
    print("Downloading ARC-AGI evaluation dataset...")
    loader = ARCLoader(cache_dir="./arc_data")

    success = loader.download_dataset("evaluation")

    if success:
        print("\nSuccessfully downloaded evaluation dataset!")
        tasks = loader.load_all_tasks("evaluation")
        print(f"Total evaluation tasks: {len(tasks)}")
    else:
        print("\nFailed to download evaluation dataset.")

if __name__ == "__main__":
    main()
