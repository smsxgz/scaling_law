import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Union

TASK_NAME = "logprobs_mixed_scaling_law"

# Features for the model (fixed effects) - 'problem_id' removed
FEATURE_NAMES = ["params", "tokens"]
TARGET_NAME = "logprob" 
# Column for random effects (mixed-effects grouping)
GROUP_COLUMN = "problem_id"
TASK_NAME_COLUMN = "task_name"

# Data files
TRAIN_FILE = "scaling_train_dataset.csv"
VALIDATION_FILE = "scaling_validation_dataset.csv"
TEST_FILE = "scaling_test_dataset.csv"

LoadedDataDict = Dict[str, Dict[str, np.ndarray]]


def load_data(train: bool = True, mode: str = "evolve") -> LoadedDataDict:
    """
    Loads data for the experiment.

    Args:
        train: If True, load training data. If False, load test/validation data.
        mode: 'evolve' (use train/validation) or 'final' (use train+val/test).

    Returns:
        A dictionary mapping task_name to its data ("data_points", "y").
    """
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
            train_data = _load_local_data(TRAIN_FILE)
            val_data = _load_local_data(VALIDATION_FILE)
            return _merge_data_dicts(train_data, val_data)
        else:
            return _load_local_data(TEST_FILE)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: evolve, final")

def _load_local_data(file_name: str) -> LoadedDataDict:
    """
    Load data from local CSV file and group it by task_name.
    """
    file_path = Path(file_name)

    # Restore robust file path check (current and parent directory)
    if not file_path.exists():
        raise FileNotFoundError(f"Local data file {file_name} not found in current or parent directory")

    # Read CSV data
    df = pd.read_csv(file_path)

    # Check required columns exist
    all_required_cols = (
        FEATURE_NAMES + 
        [TARGET_NAME, GROUP_COLUMN, TASK_NAME_COLUMN]
    )
    # Remove duplicates
    all_required_cols = sorted(list(set(all_required_cols))) 
    
    missing_cols = [col for col in all_required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {file_name}: {missing_cols}")

    processed_data: LoadedDataDict = {}
    
    # Group by task_name for evaluation
    unique_tasks = sorted(df[TASK_NAME_COLUMN].unique())

    for task_key in unique_tasks:
        # Filter data for current task
        task_data = df[df[TASK_NAME_COLUMN] == task_key].copy()
        
        if task_data.empty:
            continue

        X = task_data[FEATURE_NAMES].values.astype(float)
        
        groups = task_data[GROUP_COLUMN].values
        groups_reshaped = groups.reshape(-1, 1)
        
        data_points = np.hstack((groups_reshaped, X))

        y = task_data[TARGET_NAME].values.astype(float)

        processed_data[task_key] = {
            "X": data_points,
            "y": y
        }

    return processed_data


def _merge_data_dicts(data1: LoadedDataDict, data2: LoadedDataDict) -> LoadedDataDict:
    merged = data1.copy()
    merged.update(data2)
    return merged
