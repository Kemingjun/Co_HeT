"""Read one packaged HRSP workbook from an explicit instance directory."""

import ast
import os
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "task_index",
    "source_x",
    "source_y",
    "deadline",
    "t_operation",
    "required_robot",
]


def read_excel(file_name):
    path = Path(file_name)
    if not path.is_absolute():
        instance_dir = os.environ.get("COHET_INSTANCE_DIR")
        if not instance_dir:
            raise RuntimeError("COHET_INSTANCE_DIR is required")
        path = Path(instance_dir) / path
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_excel(path)
    if frame.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns in {path}: {frame.columns.tolist()}")
    instance = [list(row) for _, row in frame.iterrows()]
    for task_info in instance:
        task_info[-2] = ast.literal_eval(str(task_info[-2]))
        task_info[-1] = ast.literal_eval(str(task_info[-1]))
    return instance
