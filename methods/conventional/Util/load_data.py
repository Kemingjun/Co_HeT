import pandas as pd
from pathlib import Path
import numpy as np
import ast


def read_excel(file_name):
    file_name = str(Path(__file__).resolve().parent.parent) + "/Instance/" + file_name
    df = pd.read_excel(file_name)
    instance = [list(row) for index, row in df.iterrows()]
    for task_info in instance:
        required_robot = task_info[-1]
        _required_robot = ast.literal_eval(required_robot)
        task_info[-1] = _required_robot

        operation_time = task_info[-2]
        _operation_time = ast.literal_eval(operation_time)
        task_info[-2] = _operation_time
    return instance

# if __name__ == "__main__":
#     filename = '10_20250506110532.xlsx'
#     instance = read_excel(filename)
#     pass


