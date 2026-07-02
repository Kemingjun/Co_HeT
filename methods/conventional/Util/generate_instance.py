import random
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from Util.Config import Config
import math



def generate_instance(size, ROBOT_TYPE_NUM):
    instance = []

    ddl_base = [i * 0.3 + 0.3 for i in range(size)]
    noise = [random.uniform(-0.2, 0.2) for _ in range(size)]
    ddl_list = [b + n for b, n in zip(ddl_base, noise)]
    random.shuffle(ddl_list)

    location_list = []

    for task in range(1, size + 1):
        while True:
            source_x = random.random()
            source_y = random.random()
            feasible = True
            for location in location_list:
                if math.hypot(location[0] - source_x, location[1] - source_y) < 0.1:
                    feasible = False
            if feasible:
                break

        location_list.append([source_x, source_y])
        # operation_time_list = [random.uniform(0.3, 0.7), random.uniform(0.6, 1.0), random.uniform(0.8, 1.2)]
        operation_time_list = [random.uniform(0.3, 0.7), random.uniform(0.8, 1.2)]
        # operation_time = [1, 0.5]
        # operation_time = 1
        deadline = ddl_list[task - 1]
        # deadline = 6
        required_robot_list = [1 for _ in range(ROBOT_TYPE_NUM)]
        # required_robot_list = [1  for _ in range(ROBOT_TYPE_NUM)]
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

    base_dir = Path(__file__).resolve().parent.parent / "Instance"
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

        # out_name = f"T{task_num}_{ROBOT_TYPE_NUM}_I{Path(base_name).stem}.xlsx"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    df.to_excel(out_path, index=False)



if __name__ == "__main__":
    size = 50
    # for i in range(1,21):
    #     file_name = f"N{size}_K2_M12_GEN/N{size}_K2_M12_I{i}"
    ROBOT_TYPE_NUM = 2
    instance = generate_instance(size, ROBOT_TYPE_NUM)
    instance_2_excel(instance, ROBOT_TYPE_NUM, "N50_K2_M12_animation")





