import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Union

# Defines the schema for each task
TASK_SCHEMA_MAP = {
    "logprobs_mixed_scaling_law": {
        # The feature names *required by the model*.
        # "group" is the problem_id and is now a feature.
        "feature_names": ["group", "params", "tokens"],
        "target_name": "loss",
        "train_file": "logprobs_scaling_train_dataset.csv",
        "validation_file": "logprobs_scaling_validation_dataset.csv",
        "test_file": "logprobs_scaling_test_dataset.csv",
        "group_column": "group"  # The column to use for grouping
    }
}

def load_data(
    task_name: str,
    train: bool = True,
    mode: str = "evolve",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load scaling law data from local CSV file.

    The dataset is grouped by the 'group_column' key (problem_id).
    The function returns a dictionary mapping each group key to a tuple of (features, labels).

    - features (X): A numpy array of shape (n_samples, n_features).
                    For this task, n_features=3, corresponding to [problem_id, params, tokens].
                    The array will have dtype=object because it mixes strings and floats.
    - labels (y): A numpy array of shape (n_samples,).

    Args:
        task_name: The name of the task (e.g., 'logprobs_scaling_law').
        train: If True, load training data; otherwise, load test data.
        mode: Either "evolve" or "final".
              - "evolve": During evolution, train uses train_file, test uses validation_file
              - "final": For final testing, train uses train+validation merged, test uses test_file

    Returns:
        A dictionary containing the prepared data, structured by group.
    """
    if task_name not in TASK_SCHEMA_MAP:
        raise ValueError(f"Task '{task_name}' not found. Available tasks: {list(TASK_SCHEMA_MAP.keys())}")

    schema = TASK_SCHEMA_MAP[task_name]

    if mode == "evolve":
        # During evolution: use train for training, validation for testing
        if train:
            file_name = schema["train_file"]
            return _load_local_data(file_name, schema)
        else:
            file_name = schema["validation_file"]
            return _load_local_data(file_name, schema)
    elif mode == "final":
        # Final testing: merge train+validation for training, use test for testing
        if train:
            # Load and merge train and validation data
            train_data = _load_local_data(schema["train_file"], schema)
            val_data = _load_local_data(schema["validation_file"], schema)
            return _merge_data_dicts(train_data, val_data)
        else:
            file_name = schema["test_file"]
            return _load_local_data(file_name, schema)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: evolve, final")

def _load_local_data(
    file_name: str,
    schema: Dict[str, Union[str, List[str]]]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
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

    feature_names = schema["feature_names"]
    target_name = schema["target_name"]
    group_column = schema.get("group_column", "group")

    # Check required columns exist
    all_required_cols = list(feature_names) + [target_name, group_column]
    # Remove duplicates if any (e.g., group_column is in feature_names)
    all_required_cols = sorted(list(set(all_required_cols)))
    
    missing_cols = [col for col in all_required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {file_name}: {missing_cols}")

    processed_data = {}
    unique_groups = sorted(df[group_column].unique())

    for group_key in unique_groups:
        # Filter data for current group
        group_data = df[df[group_column] == group_key].copy()
        
        if group_data.empty:
            continue

        # Extract features
        # This will create an np.ndarray with dtype=object
        # e.g., [['mmlu_sec_97', 1.4e9, 3.0e11], ['mmlu_sec_97', 1.2e10, 3.0e11]]
        X = group_data[feature_names].values

        # Extract targets
        y = group_data[target_name].values.astype(float)
        
        processed_data[group_key] = (X, y)

    return processed_data


def _merge_data_dicts(
    data1: Dict[str, Tuple[np.ndarray, np.ndarray]],
    data2: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Merge two data dictionaries.

    Since train and validation datasets have no overlapping groups,
    we simply combine them into one dictionary.

    Args:
        data1: First data dictionary (train)
        data2: Second data dictionary (validation)

    Returns:
        Merged data dictionary containing all groups from both datasets
    """
    merged = data1.copy()
    merged.update(data2)
    return merged