"""
ARC dataset loader and task representation.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import requests
from pathlib import Path


@dataclass
class ARCTask:
    """
    Represents a single ARC task.

    Attributes:
        task_id: Unique identifier
        train: List of (input_grid, output_grid) pairs for training
        test: List of (input_grid, output_grid) pairs for testing
    """
    task_id: str
    train: List[Tuple[np.ndarray, np.ndarray]]
    test: List[Tuple[np.ndarray, np.ndarray]]

    @property
    def num_train(self) -> int:
        """Number of training examples."""
        return len(self.train)

    @property
    def num_test(self) -> int:
        """Number of test examples."""
        return len(self.test)

    def get_train_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a specific training pair."""
        return self.train[idx]

    def get_test_input(self, idx: int) -> np.ndarray:
        """Get a specific test input."""
        return self.test[idx][0]

    def get_test_output(self, idx: int) -> np.ndarray:
        """Get the ground truth test output."""
        return self.test[idx][1]


class ARCLoader:
    """
    Loader for ARC-AGI dataset.
    """

    BASE_URL = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"

    def __init__(self, cache_dir: str = "./arc_data"):
        """
        Initialize the ARC loader.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tasks: Dict[str, ARCTask] = {}

    def download_dataset(self, split: str = "training") -> bool:
        """
        Download ARC dataset from GitHub.

        Args:
            split: Dataset split ('training', 'evaluation', 'test')

        Returns:
            True if successful
        """
        url = f"{self.BASE_URL}/{split}"
        local_dir = self.cache_dir / split

        # Check if already exists
        if local_dir.exists() and len(list(local_dir.glob("*.json"))) > 0:
            print(f"Dataset {split} already exists in {local_dir}")
            return True

        local_dir.mkdir(parents=True, exist_ok=True)

        # Download index to get file list
        try:
            # Try to get a known task to verify connection
            test_url = f"{url}/0d3d703e.json"
            response = requests.get(test_url, timeout=10)

            if response.status_code == 200:
                # Download a few sample tasks
                sample_ids = [
                    "0d3d703e", "1e0a9b12", "25ff71a9", "3c9b0459", "6150a2bd",
                    "007bbfb7", "00d62c1b", "017c7c7b", "025d127b", "045e512c"
                ]

                for task_id in sample_ids:
                    task_url = f"{url}/{task_id}.json"
                    try:
                        resp = requests.get(task_url, timeout=10)
                        if resp.status_code == 200:
                            file_path = local_dir / f"{task_id}.json"
                            with open(file_path, 'w') as f:
                                f.write(resp.text)
                            print(f"Downloaded {task_id}.json")
                    except Exception as e:
                        print(f"Failed to download {task_id}: {e}")

                return True

        except Exception as e:
            print(f"Failed to download dataset: {e}")
            return False

        return False

    def load_task(self, file_path: str) -> ARCTask:
        """
        Load a single ARC task from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            ARCTask object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract task ID from filename
        task_id = Path(file_path).stem

        # Parse training examples
        train = []
        for example in data.get('train', []):
            input_grid = np.array(example['input'], dtype=np.int32)
            output_grid = np.array(example['output'], dtype=np.int32)
            train.append((input_grid, output_grid))

        # Parse test examples
        test = []
        for example in data.get('test', []):
            input_grid = np.array(example['input'], dtype=np.int32)
            # Test output might not always be available
            output_grid = np.array(example.get('output', example['input']), dtype=np.int32)
            test.append((input_grid, output_grid))

        return ARCTask(task_id=task_id, train=train, test=test)

    def load_all_tasks(self, split: str = "training") -> Dict[str, ARCTask]:
        """
        Load all tasks from a split.

        Args:
            split: Dataset split to load

        Returns:
            Dictionary mapping task IDs to ARCTask objects
        """
        split_dir = self.cache_dir / split

        if not split_dir.exists():
            print(f"Dataset {split} not found. Attempting to download...")
            self.download_dataset(split)

        tasks = {}
        json_files = list(split_dir.glob("*.json"))

        for file_path in json_files:
            try:
                task = self.load_task(str(file_path))
                tasks[task.task_id] = task
                self.tasks[task.task_id] = task
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

        print(f"Loaded {len(tasks)} tasks from {split}")
        return tasks

    def get_task(self, task_id: str) -> ARCTask:
        """Get a specific task by ID."""
        if task_id not in self.tasks:
            # Try to load it
            for split in ["training", "evaluation", "test"]:
                file_path = self.cache_dir / split / f"{task_id}.json"
                if file_path.exists():
                    task = self.load_task(str(file_path))
                    self.tasks[task_id] = task
                    return task

        return self.tasks.get(task_id)

    def get_random_task(self) -> ARCTask:
        """Get a random task from loaded tasks."""
        import random
        if not self.tasks:
            self.load_all_tasks()

        task_id = random.choice(list(self.tasks.keys()))
        return self.tasks[task_id]
