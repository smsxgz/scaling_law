import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Union

# Defines the schema for each task
TASK_SCHEMA_MAP = {
    "logprobs_scaling_law": {
        "feature_names": ["group", "params", "tokens"],
        "target_name": "loss",
        "train_file": "logprobs_scaling_train_dataset.csv",
        "test_file": "logprobs_scaling_test_dataset.csv",
        "group_column": "group"
    }
}

def load_data(
    task_name: str,
    train: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load scaling law data from local CSV file.

    The dataset is grouped by the 'group_column' key (problem_id). 
    The function returns a dictionary mapping each group key to a tuple of (features, labels).
    
    - features (X): (n_samples, 3) array [problem_id, params, tokens]
    - labels (y): (n_samples,) array
    """
    if task_name not in TASK_SCHEMA_MAP:
        raise ValueError(f"Task '{task_name}' not found. Available tasks: {list(TASK_SCHEMA_MAP.keys())}")

    schema = TASK_SCHEMA_MAP[task_name]
    file_name = schema["train_file"] if train else schema["test_file"]
    
    return _load_local_data(file_name, schema)


def _load_local_data(
    file_name: str,
    schema: Dict[str, Union[str, List[str]]]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load data from local CSV file into a dictionary grouped by 'group_column'."""
    file_path = Path(file_name)
    if not file_path.exists():
        parent_file_path = Path("..") / file_name
        if parent_file_path.exists():
            file_path = parent_file_path
        else:
            raise FileNotFoundError(f"Local data file {file_name} not found in current or parent directory")

    df = pd.read_csv(file_path)

    feature_names = schema["feature_names"]
    target_name = schema["target_name"]
    group_column = schema.get("group_column", "group")

    # Check required columns exist
    all_required_cols = sorted(list(set(list(feature_names) + [target_name, group_column])))
    missing_cols = [col for col in all_required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {file_name}: {missing_cols}")

    processed_data = {}
    unique_groups = sorted(df[group_column].unique())

    for group_key in unique_groups:
        group_data = df[df[group_column] == group_key].copy()
        if group_data.empty:
            continue
        
        X = group_data[feature_names].values
        y = group_data[target_name].values.astype(float)
        
        processed_data[group_key] = (X, y)

    return processed_data