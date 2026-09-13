import math

import numpy as np


def replay_schedule(
    source,
    deadline,
    operation_time,
    tour,
    robot_num_list,
    velocity=1.0,
    distance_weight=0.5,
):
    source = np.asarray(source, dtype=np.float32)
    deadline = np.asarray(deadline, dtype=np.float32)
    operation_time = np.asarray(operation_time, dtype=np.float32)
    tour = np.asarray(tour, dtype=np.int64).reshape(-1)
    robot_num_list = tuple(int(value) for value in robot_num_list)
    task_count = source.shape[0]
    robot_type_count = len(robot_num_list)
    action_width = 1 + robot_type_count

    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source must have shape [n, 2]")
    if deadline.shape != (task_count,):
        raise ValueError("deadline must have shape [n]")
    if operation_time.shape != (task_count, robot_type_count):
        raise ValueError("operation_time must have shape [n, robot_type_count]")
    if tour.size != task_count * action_width:
        raise ValueError("tour length does not match n * (1 + robot_type_count)")
    if velocity <= 0 or not 0 <= distance_weight <= 1:
        raise ValueError("invalid velocity or distance weight")

    current_time = np.zeros(sum(robot_num_list), dtype=np.float32)
    current_coord = np.zeros((sum(robot_num_list), 2), dtype=np.float32)
    robot_distance = np.zeros(sum(robot_num_list), dtype=np.float32)
    total_tardiness = np.float32(0.0)
    offsets = np.cumsum((0,) + robot_num_list)
    visited = set()

    for action in tour.reshape(task_count, action_width):
        task = int(action[0])
        if task < 0 or task >= task_count or task in visited:
            raise ValueError("tour must visit each task exactly once")
        robots = [int(value) for value in action[1:]]
        for robot_type, robot in enumerate(robots):
            if robot < offsets[robot_type] or robot >= offsets[robot_type + 1]:
                raise ValueError("robot index does not match its robot type")

        task_coord = source[task]
        distances = [
            np.float32(np.abs(task_coord - current_coord[robot]).sum(dtype=np.float32))
            for robot in robots
        ]
        arrivals = [
            np.float32(current_time[robot] + distance / np.float32(velocity))
            for robot, distance in zip(robots, distances)
        ]
        synchronized_start = np.float32(max(arrivals))
        completion_times = []
        for robot_type, robot in enumerate(robots):
            robot_distance[robot] = np.float32(robot_distance[robot] + distances[robot_type])
            current_coord[robot] = task_coord
            current_time[robot] = np.float32(synchronized_start + operation_time[task, robot_type])
            completion_times.append(current_time[robot])
        completion = np.float32(max(completion_times))
        total_tardiness = np.float32(
            total_tardiness + np.maximum(completion - deadline[task], np.float32(0.0))
        )
        visited.add(task)

    total_distance = np.sum(robot_distance, dtype=np.float32)
    objective = np.float32(
        np.float32(distance_weight) * total_distance
        + np.float32(1.0 - distance_weight) * total_tardiness
    )
    values = (float(total_distance), float(total_tardiness), float(objective))
    if len(visited) != task_count or not all(math.isfinite(value) for value in values):
        raise ValueError("invalid replay result")
    return {"distance": values[0], "tardiness": values[1], "objective": values[2]}
