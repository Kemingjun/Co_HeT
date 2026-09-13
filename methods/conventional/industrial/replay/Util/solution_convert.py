from Util.Solution import Solution
from Util.util import Config, path_map2sequence_map


def complete_path_map(path_map):
    return {robot: list(path_map.get(robot, [])) for robot in range(1, Config.ROBOT_NUM + 1)}


def path_map_to_solution(instance, path_map):
    full_path_map = complete_path_map(path_map)
    sequence_map, path_init_task_map = path_map2sequence_map(full_path_map)
    return Solution(instance, sequence_map, path_init_task_map)


def path_map_to_action_sequence(path_map):
    full_path_map = complete_path_map(path_map)
    task_robot = {}
    remaining_predecessors = {}
    successors = {}

    for robot, path in full_path_map.items():
        for idx, task in enumerate(path):
            task_robot.setdefault(task, {})[_robot_type(robot)] = robot
            remaining_predecessors.setdefault(task, set())
            successors.setdefault(task, [])
            if idx > 0:
                pre_task = path[idx - 1]
                remaining_predecessors[task].add(pre_task)
                successors.setdefault(pre_task, []).append(task)

    ready = sorted([task for task, pre in remaining_predecessors.items() if not pre])
    actions = []
    while ready:
        task = ready.pop(0)
        robot_map = task_robot[task]
        actions.append(
            {
                "task": task,
                "carrier": robot_map[Config.CARRIER_TYPE],
                "shuttle": robot_map[Config.SHUTTLE_TYPE],
                "forklift": robot_map[Config.FORKLIFT_TYPE],
            }
        )
        for next_task in successors.get(task, []):
            remaining_predecessors[next_task].discard(task)
            if not remaining_predecessors[next_task] and next_task not in ready:
                ready.append(next_task)
        ready.sort()

    if len(actions) != len(task_robot):
        raise ValueError("path_map contains cyclic or incomplete task precedence")
    return actions


def action_sequence_to_path_map(actions, zero_based=False):
    path_map = {robot: [] for robot in range(1, Config.ROBOT_NUM + 1)}
    task_seen = set()
    for action in actions:
        if isinstance(action, dict):
            task = action["task"]
            carrier = action["carrier"]
            shuttle = action["shuttle"]
            forklift = action["forklift"]
        else:
            task, carrier, shuttle, forklift = action[:4]

        if zero_based:
            task += 1
            carrier += 1
            shuttle += 1
            forklift += 1

        task = int(task)
        robots = [int(carrier), int(shuttle), int(forklift)]
        _validate_action(task, robots)
        if task in task_seen:
            raise ValueError(f"duplicated task in action_sequence: {task}")
        task_seen.add(task)
        for robot in robots:
            path_map[robot].append(task)
    return path_map


def pi_to_path_map(pi):
    if hasattr(pi, "detach"):
        pi = pi.detach().cpu().tolist()
    return action_sequence_to_path_map(pi, zero_based=True)


def _robot_type(robot):
    for robot_type, (low, high) in enumerate(Config.TYPE_LIST, start=1):
        if low <= robot < high:
            return robot_type
    raise ValueError(f"invalid robot id: {robot}")


def _validate_action(task, robots):
    if task <= 0:
        raise ValueError(f"task ids must be 1-based positive integers, got {task}")
    expected_types = [Config.CARRIER_TYPE, Config.SHUTTLE_TYPE, Config.FORKLIFT_TYPE]
    actual_types = [_robot_type(robot) for robot in robots]
    if actual_types != expected_types:
        raise ValueError(f"expected carrier/shuttle/forklift robots, got ids {robots}")
