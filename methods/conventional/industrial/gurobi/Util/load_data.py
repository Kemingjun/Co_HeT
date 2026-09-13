import pandas as pd
from pathlib import Path
import ast
import os

from Util.RealWorldConfig import RealWorldConfig


REAL_WORLD_COLUMNS = [
    RealWorldConfig.TASK_INDEX_COL,
    RealWorldConfig.SUPPLY_X_COL,
    RealWorldConfig.SUPPLY_Y_COL,
    RealWorldConfig.HANDOVER_X_COL,
    RealWorldConfig.HANDOVER_Y_COL,
    RealWorldConfig.DELIVERY_X_COL,
    RealWorldConfig.DELIVERY_Y_COL,
    RealWorldConfig.DEADLINE_COL,
]


def read_excel(file_name):
    default_instance_dir = Path(__file__).resolve().parents[1] / "instances"
    instance_dir = Path(os.environ.get("COHET_INDUSTRIAL_INSTANCE_DIR", default_instance_dir)).resolve()
    file_path = Path(file_name)
    if not file_path.is_absolute():
        file_path = instance_dir / file_path.name
    df = pd.read_excel(file_path)
    if list(df.columns) == REAL_WORLD_COLUMNS:
        instance = [list(row) + [[1 for _ in range(RealWorldConfig.ROBOT_TYPE_NUM)]] for _, row in df.iterrows()]
        return instance

    instance = [list(row) for index, row in df.iterrows()]
    for task_info in instance:
        if isinstance(task_info[-1], str):
            task_info[-1] = ast.literal_eval(task_info[-1])
        if isinstance(task_info[-2], str):
            task_info[-2] = ast.literal_eval(task_info[-2])
    return instance

# if __name__ == "__main__":
#     filename = '10_20250506110532.xlsx'
#     instance = read_excel(filename)
#     pass
