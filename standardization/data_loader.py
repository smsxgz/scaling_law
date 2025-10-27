import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Union

# Task Schema
TASK_NAME = "logprobs_mixed_scaling_law"
FEATURE_NAMES = ["params", "tokens"]
TARGET_NAME = "logprob_zscore"

TRAIN_FILE = "scaling_train_dataset.csv"
VALIDATION_FILE = "scaling_validation_dataset.csv"
TEST_FILE = "scaling_test_dataset.csv"


def load_data(train: bool = True, mode: str = "evolve") -> Tuple[np.ndarray, np.ndarray]:
    if mode == "evolve":
        # During evolution: use train for training, validation for testing
        if train:
            return _load_local_data(TRAIN_FILE)
        else:
            return _load_local_data(VALIDATION_FILE)
    
    elif mode == "final":
        # Final testing: merge train+validation for training, use test for testing
        if train:
            # Load and merge train and validation data
            X_train, y_train = _load_local_data(TRAIN_FILE)
            X_val, y_val = _load_local_data(VALIDATION_FILE)
            
            X_merged = np.vstack((X_train, X_val))
            y_merged = np.concatenate((y_train, y_val))
            
            return X_merged, y_merged
        else:
            return _load_local_data(TEST_FILE)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: evolve, final")

def _load_local_data(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from local CSV file."""
    file_path = Path(file_name)

    # Try to find the file in current directory or parent directory
    if not file_path.exists():
        parent_file_path = Path("..") / file_name
        if parent_file_path.exists():
            file_path = parent_file_path
        else:
            raise FileNotFoundError(f"Local data file {file_name} not found in current or parent directory")

    # Read CSV data
    df = pd.read_csv(file_path)

    # Check required columns exist
    all_required_cols = FEATURE_NAMES + [TARGET_NAME]
    
    missing_cols = [col for col in all_required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {file_name}: {missing_cols}")

    X = df[FEATURE_NAMES].values.astype(float)
    y = df[TARGET_NAME].values.astype(float)

    return X, y