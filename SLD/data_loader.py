"""
Unified data loading interface for scaling law discovery.

Dynamically loads data from the Hugging Face Hub repository 'pkuHaowei/sldbench'.
Also supports local data loading for custom datasets.
This approach centralizes data access and ensures consistency.
"""
import numpy as np
import datasets
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, Tuple

# --- Configuration ---

HUB_REPO_ID = "pkuHaowei/sldbench"

# Defines the schema for each task, mapping feature/target names from the Hub
# to the columns in the dataset.
TASK_SCHEMA_MAP = {
    "data_constrained_scaling_law": {
        "feature_names": ["unique_tokens", "params", "tokens"],
        "target_name": "loss",
    },
    "domain_mixture_scaling_law": {
        "feature_names": [f"proportion_domain_{i+1}" for i in range(5)],
        "target_name": [f"loss_domain_{i+1}" for i in range(5)],
    },
    "lr_bsz_scaling_law": {
        "feature_names": ["lr", "bsz", "data_size", "non_embedding_param_size"],
        "target_name": "lm_loss",
    },
    "moe_scaling_law": {
        "feature_names": ["num_experts", "dense_parameter_count"],
        "target_name": "loss_validation",
    },
    "sft_scaling_law": {
        "feature_names": ["sft_data_size"],
        "target_name": "sft_loss",
    },
    "vocab_scaling_law": {
        "feature_names": ["non_vocab_parameters", "vocab_size", "num_characters"],
        "target_name": "unigram_normalized_loss",
    },
    "parallel_scaling_law": {
        "feature_names": ["num_params", "parallel_size"],
        "target_name": "loss"
    },
    # Local tasks
    "logprobs_scaling_law": {
        "feature_names": ["params", "tokens"],
        "target_name": "loss",
        "local_file": "logprobs_scaling_dataset.csv",
        "group_column": "group"
    }
}

def load_data(
    app_name: str,
    train: bool = True,
) -> Dict[Any, Tuple[np.ndarray, np.ndarray]]:
    """
    Unified data loading interface. Loads and processes data from Hugging Face Hub
    or local files.

    Each task's dataset is grouped by a 'group' key. The function returns a
    dictionary mapping each group key to a tuple of (features, labels).
    - features (X): A numpy array of shape (n_samples, n_features).
    - labels (y): A numpy array of shape (n_samples,) or (n_samples, n_targets).

    Args:
        app_name: The name of the task (e.g., 'sft_scaling_law').
        train: If True, load training data; otherwise, load test data.

    Returns:
        A dictionary containing the prepared data, structured by group.
    """
    if app_name not in TASK_SCHEMA_MAP:
        raise ValueError(f"Task '{app_name}' not found in TASK_SCHEMA_MAP. Available tasks: {list(TASK_SCHEMA_MAP.keys())}")

    schema = TASK_SCHEMA_MAP[app_name]

    # Check if this is a local task
    if "local_file" in schema:
        return _load_local_data(app_name, train, schema)
    else:
        return _load_hub_data(app_name, train, schema)


def _load_local_data(
    app_name: str,
    train: bool,
    schema: Dict[str, Any]
) -> Dict[Any, Tuple[np.ndarray, np.ndarray]]:
    """Load data from local CSV file."""
    local_file = schema["local_file"]
    file_path = Path(local_file)

    # Try to find the file in current directory or parent directory
    if not file_path.exists():
        parent_file_path = Path("..") / local_file
        if parent_file_path.exists():
            file_path = parent_file_path
        else:
            raise FileNotFoundError(f"Local data file {local_file} not found in current or parent directory")

    # Read CSV data
    df = pd.read_csv(file_path)

    feature_names = schema["feature_names"]
    target_name = schema["target_name"]
    group_column = schema.get("group_column", "group")

    # Check required columns exist
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in {local_file}: {missing_features}")

    if target_name not in df.columns:
        raise ValueError(f"Missing target column '{target_name}' in {local_file}")

    if group_column not in df.columns:
        raise ValueError(f"Missing group column '{group_column}' in {local_file}")

    processed_data = {}
    unique_groups = sorted(df[group_column].unique())

    for group_key in unique_groups:
        # Filter data for current group
        group_data = df[df[group_column] == group_key].copy()

        # Extract features
        X = group_data[feature_names].values.astype(float)

        # Extract targets
        y = group_data[target_name].values.astype(float)

        # Simple train/test split (70%/30%)
        n_samples = len(group_data)
        n_train = int(0.7 * n_samples)

        if train:
            X_split = X[:n_train]
            y_split = y[:n_train]
        else:
            X_split = X[n_train:]
            y_split = y[n_train:]

        # Skip empty splits
        if len(X_split) == 0:
            continue

        processed_data[group_key] = (X_split, y_split)

    return processed_data


def _load_hub_data(
    app_name: str,
    train: bool,
    schema: Dict[str, Any]
) -> Dict[Any, Tuple[np.ndarray, np.ndarray]]:
    """Load data from Hugging Face Hub."""
    split = 'train' if train else 'test'

    try:
        # Load the specific task dataset from the Hugging Face Hub
        dataset = datasets.load_dataset(HUB_REPO_ID, name=app_name, split=split)
    except Exception as e:
        raise IOError(f"Failed to load dataset '{app_name}' with split '{split}' from '{HUB_REPO_ID}'. Reason: {e}")

    # Ensure target_name is a list for consistent processing
    feature_names = schema["feature_names"]
    target_names = schema["target_name"]
    if not isinstance(target_names, list):
        target_names = [target_names]

    processed_data = {}

    # The datasets are partitioned by a 'group' column
    unique_groups = sorted(list(set(dataset['group'])))

    for group_key in unique_groups:
        # Filter the dataset for the current group
        group_data = dataset.filter(lambda example: example['group'] == group_key)

        # Extract features (X) and stack them into a single numpy array
        X_list = [np.array(group_data[fname]) for fname in feature_names]
        X = np.stack(X_list, axis=1)

        # Extract targets (y)
        y_list = [np.array(group_data[tname]) for tname in target_names]
        y_stacked = np.stack(y_list, axis=1)

        # Squeeze the last dimension if there is only one target
        y = y_stacked.squeeze(axis=1) if y_stacked.shape[1] == 1 else y_stacked

        processed_data[group_key] = (X, y)

    return processed_data

if __name__ == '__main__':
    # Example of how to use the new loader
    # The list of tasks is now derived directly from the schema map
    ALL_TASKS = list(TASK_SCHEMA_MAP.keys())

    for task in ALL_TASKS:
        print(f"\n--- Testing '{task}' ---")
        try:
            # Load training data
            train_data = load_data(task, train=True)
            print(f"Successfully loaded training data from Hugging Face repo '{HUB_REPO_ID}'.")
            
            # Inspect the first group's shape
            first_group_key = next(iter(train_data))
            X_train, y_train = train_data[first_group_key]
            print(f"Train groups: {len(train_data)}. First group '{first_group_key}' shape: X={X_train.shape}, y={y_train.shape}")
            
            # Load test data
            test_data = load_data(task, train=False)
            if test_data:
                first_test_key = next(iter(test_data))
                X_test, y_test = test_data[first_test_key]
                print(f"Test groups: {len(test_data)}. First group '{first_test_key}' shape: X={X_test.shape}, y={y_test.shape}")
            else:
                print("Test data is empty.")

        except (ValueError, IOError, KeyError) as e:
            print(f"Error loading data for task '{task}': {e}")