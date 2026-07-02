import random
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from Util.Config import Config


def generate_points(size, dist_type="uniform", seed=None):
    if seed is not None:
        np.random.seed(seed)

    if dist_type == "uniform":
        x = np.random.rand(size)
        y = np.random.rand(size)
        return np.column_stack([x, y])

    elif dist_type == "gaussian":
        centers = np.array([[0.3, 0.3], [0.7, 0.7]])
        sigma = 0.09
        points = np.vstack([
            np.random.normal(loc=center, scale=sigma, size=(size // 2, 2))
            for center in centers
        ])
        return np.clip(points, 0, 1)

    elif dist_type == "annulus":
        r_inner, r_outer = 0.25, 0.5
        theta = np.random.uniform(0, 2 * np.pi, size)
        r = np.random.uniform(r_inner, r_outer, size)
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        return np.clip(np.column_stack([x, y]), 0, 1)

    elif dist_type == "semicircle":
        r = np.sqrt(np.random.uniform(0, 0.5 ** 2, size))
        theta = np.random.uniform(0, np.pi, size)
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        return np.clip(np.column_stack([x, y]), 0, 1)

    elif dist_type == "stripe":
        x = np.random.uniform(0, 1, size)
        y = np.random.normal(0.5, 0.1, size)
        return np.column_stack([x, np.clip(y, 0, 1)])

    else:
        raise ValueError("Unknown distribution type.")


def generate_instance(size, ROBOT_TYPE_NUM, dist_type="uniform"):
    instance = []
    ddl_base = [i * 0.5 + 0.5 for i in range(size)]
    noise = [random.uniform(-0.2, 0.2) for _ in range(size)]
    ddl_list = [b + n for b, n in zip(ddl_base, noise)]
    random.shuffle(ddl_list)

    points = generate_points(size, dist_type=dist_type)

    for task in range(1, size + 1):
        source_x, source_y = points[task - 1]
        operation_time_list = [random.uniform(0.3, 0.7), random.uniform(0.8, 1.2)]
        deadline = ddl_list[task - 1]
        required_robot_list = [1 for _ in range(ROBOT_TYPE_NUM)]
        if sum(required_robot_list) == 0:
            required_robot_list[random.randint(0, len(required_robot_list) - 1)] += 1

        instance.append([task, source_x, source_y, deadline, operation_time_list, required_robot_list])
    return instance


def instance_2_excel(instance, ROBOT_TYPE_NUM, filename=None):
    task_index, source_x, source_y, deadline, t_operation, required_robot = [], [], [], [], [], []
    for task_info in instance:
        task_index.append(task_info[0])
        source_x.append(task_info[1])
        source_y.append(task_info[2])
        deadline.append(task_info[3])
        t_operation.append(task_info[4])
        required_robot.append(task_info[5])

    df = pd.DataFrame({
        'task_index': task_index,
        'source_x': source_x,
        'source_y': source_y,
        'deadline': deadline,
        't_operation': t_operation,
        'required_robot': required_robot
    })

    base_dir = Path(__file__).resolve().parent.parent / "Instance_robustness"
    task_num = len(instance)

    if not filename:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        out_dir = base_dir
        out_name = f"T{task_num}_{ROBOT_TYPE_NUM}_{ts}.xlsx"
    else:
        f = Path(filename)
        out_dir = base_dir / f.parent
        base_name = f.name
        out_name = f"{Path(base_name).stem}.xlsx"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    df.to_excel(out_path, index=False)


if __name__ == "__main__":
    size = 50
    ROBOT_TYPE_NUM = 2
    for dist in ["gaussian", "annulus", "semicircle", "stripe"]:
        for i in range(1, 21):
            file_name = f"N{size}_K{ROBOT_TYPE_NUM}_M12_{dist}/N{size}_K{ROBOT_TYPE_NUM}_M12_{dist}_I{i}"
            instance = generate_instance(size, ROBOT_TYPE_NUM, dist_type=dist)
            instance_2_excel(instance, ROBOT_TYPE_NUM, file_name)


