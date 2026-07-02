from Util.Config import Config
import bisect
import random
import math
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, filename='info.log')
def copy_set_int(original_set):
    """
    set
    setint
    :param original_set:
    :return:
    """
    copied_set = set()
    for item in original_set:
        copied_set.add(item)

    return copied_set


def copy_list_int(original_list):
    """
    set
    setint
    :param original_list:
    :return:
    """
    copied_list = []
    for item in original_list:
        copied_list.append(item)

    return copied_list


def copy_dict_int_int(original_dict):
    copied_dict = {}

    for key, value in original_dict.items():
        copied_dict[key] = value

    return copied_dict


def copy_dict_int_list(original_dict):
    copied_dict = {}

    for key, value in original_dict.items():
        copied_list = []
        for v in value:
            copied_list.append(v)
        copied_dict[key] = copied_list

    return copied_dict


def copy_dict_int_dict(original_dict):
    copied_dict = {}

    for key, inner_dict in original_dict.items():
        copied_inner_dict = {}

        for inner_key, value in inner_dict.items():
            copied_inner_dict[inner_key] = value

        copied_dict[key] = copied_inner_dict

    return copied_dict



def code2path_map(code):
    path_map = {}
    path = []
    path_index = 1
    for task_index, task in enumerate(code):
        if task == 0:
            if task_index != 0:
                path_map[path_index] = path
                path = []
                path_index += 1
                # if task_index == len(code) - 1:
                #     path_map[path_index] = []
            else:
                continue
        else:
            path.append(task)
    path_map[path_index] = path

    return path_map

def path_map2sequence_map(path_map):
    sequence_map = {}
    path_init_task_map = {}
    """
    robotrobotrobot
    """
    for path_index in range(1, Config.ROBOT_NUM + 1):
        path_type = bisect.bisect_left(Config.INDEX_LIST, path_index) + 1
        path = path_map[path_index]
        if len(path) == 0:
            path_init_task_map[path_index] = 0
        for task_index, task in enumerate(path):
            if task not in sequence_map.keys():
                task_info = {}
            else:
                task_info = sequence_map[task]
            task_info[f'robot_{path_type}'] = path_index
            if task_index == 0:
                task_info[f'robot_{path_type}_pre_task'] = 0
                path_init_task_map[path_index] = task
            else:
                task_info[f'robot_{path_type}_pre_task'] = path[task_index - 1]
            if task_index == len(path) - 1:
                task_info[f'robot_{path_type}_next_task'] = 0
            else:
                task_info[f'robot_{path_type}_next_task'] = path[task_index + 1]

            sequence_map[task] = task_info
    return sequence_map, path_init_task_map


def get_distance(source_position, destination_position):
    # distance = abs(source_position[0] - destination_position[0]) + abs(source_position[1] - destination_position[1])
    distance = math.hypot(source_position[0] - destination_position[0], source_position[1] - destination_position[1])
    return distance


def remove_(sequence_map, path_init_task_map, task):
    for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
        if f'robot_{robot_type}' not in sequence_map[task].keys():
            continue
        pre_task = sequence_map[task][f'robot_{robot_type}_pre_task']
        next_task = sequence_map[task][f'robot_{robot_type}_next_task']

        if pre_task == 0:
            robot = sequence_map[task][f'robot_{robot_type}']
            path_init_task_map[robot] = next_task
        else:
            sequence_map[pre_task][f'robot_{robot_type}_next_task'] = next_task

        if next_task != 0:
            sequence_map[next_task][f'robot_{robot_type}_pre_task'] = pre_task

    del sequence_map[task]

    return sequence_map, path_init_task_map

def insert_(sequence_map, path_init_task_map, task, position, robot_type):

    if position[0] != 0:
        neighbor_task = position[0]
        direction = ['pre', 'next'][position[1]]
        direction_opposite = ['pre', 'next'][position[1] - 1]
        direction_task = sequence_map[neighbor_task][f'robot_{robot_type}_{direction}_task']
        robot = sequence_map[neighbor_task][f'robot_{robot_type}']
        sequence_map[neighbor_task][f'robot_{robot_type}_{direction}_task'] = task

        if direction_task != 0:
            sequence_map[direction_task][f'robot_{robot_type}_{direction_opposite}_task'] = task
        else:
            if direction == 'pre':
                path_init_task_map[robot] = task
        if task not in sequence_map.keys():
            sequence_map[task] = {}
            sequence_map[task][f'robot_{robot_type}'] = robot
            sequence_map[task][f'robot_{robot_type}_{direction}_task'] = direction_task
            sequence_map[task][f'robot_{robot_type}_{direction_opposite}_task'] = neighbor_task
        else:
            sequence_map[task][f'robot_{robot_type}'] = robot
            sequence_map[task][f'robot_{robot_type}_{direction}_task'] = direction_task
            sequence_map[task][f'robot_{robot_type}_{direction_opposite}_task'] = neighbor_task
    else:
        robot = position[1]
        path_init_task_map[robot] = task
        if task not in sequence_map.keys():
            sequence_map[task] = {}
            sequence_map[task][f'robot_{robot_type}'] = robot
            sequence_map[task][f'robot_{robot_type}_pre_task'] = 0
            sequence_map[task][f'robot_{robot_type}_next_task'] = 0
        else:
            sequence_map[task][f'robot_{robot_type}'] = robot
            sequence_map[task][f'robot_{robot_type}_pre_task'] = 0
            sequence_map[task][f'robot_{robot_type}_next_task'] = 0

    return sequence_map, path_init_task_map

def get_all_position(sequence_map, path_init_task_map, robot_type):
    tasks_with_robot_type = [
        task_id
        for task_id, task_info in sequence_map.items()
        if f"robot_{robot_type}" in task_info.keys()
    ]

    position_set = {(task, 0) for task in tasks_with_robot_type}
    position_set = position_set.union({(task, 1) for task in tasks_with_robot_type})
    to_delete_list = []
    for position in position_set:
        task = position[0]
        direction = position[1]
        if direction == 0:
            pre_task = sequence_map[task][f'robot_{robot_type}_pre_task']
            if pre_task != 0 and pre_task is not None:
                if (pre_task, 1) in position_set:
                    to_delete_list.append((pre_task, 1))
        else:
            next_task = sequence_map[task][f'robot_{robot_type}_next_task']
            if next_task != 0 and next_task is not None:
                if (next_task, 0) in position_set:
                    to_delete_list.append((task, 1))

    for position in to_delete_list:
        position_set.discard(position)

    for robot, init_task in path_init_task_map.items():
        _robot_type = bisect.bisect_left(Config.INDEX_LIST, robot) + 1
        if _robot_type == robot_type and init_task == 0:
            position_set.add((0, robot))

    return position_set

def get_feasible_insert_position(sequence_map, path_init_task_map, destroyed_task, robot_type):
    feasible_position_set = get_all_position(sequence_map, path_init_task_map, robot_type)
    if destroyed_task not in sequence_map.keys():
        return feasible_position_set
    pre_to_explore_set = {destroyed_task}
    pre_explored_list = []
    while len(pre_to_explore_set) != 0:
        temp_set = copy_set_int(pre_to_explore_set)
        for task in temp_set:
            for to_explore_robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
                if f"robot_{to_explore_robot_type}" in sequence_map[task].keys():
                    pre_task = sequence_map[task][f"robot_{to_explore_robot_type}_pre_task"]
                    if pre_task != 0:
                        feasible_position_set.discard((pre_task, 0))
                        if pre_task not in pre_explored_list:
                            pre_to_explore_set.add(pre_task)
                        if f"robot_{robot_type}" in sequence_map[pre_task].keys():
                            pre_pre_task = sequence_map[pre_task][f"robot_{robot_type}_pre_task"]
                            if pre_pre_task != 0:
                                feasible_position_set.discard((pre_pre_task, 1))


            pre_to_explore_set.remove(task)
            pre_explored_list.append(task)

    next_to_explore_set = {destroyed_task}
    next_explored_list = []
    while len(next_to_explore_set) != 0:
        temp_set = copy_set_int(next_to_explore_set)
        for task in temp_set:
            for to_explore_robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
                if f"robot_{to_explore_robot_type}" in sequence_map[task].keys():
                    next_task = sequence_map[task][f"robot_{to_explore_robot_type}_next_task"]
                    if next_task != 0:
                        feasible_position_set.discard((next_task, 1))
                        if next_task not in next_explored_list:
                            next_to_explore_set.add(next_task)
                        if f"robot_{robot_type}" in sequence_map[next_task].keys():
                            next_next_task = sequence_map[next_task][f"robot_{robot_type}_next_task"]
                            if next_next_task != 0:
                                feasible_position_set.discard((next_next_task, 0))

            next_to_explore_set.remove(task)
            next_explored_list.append(task)

    return feasible_position_set



# def get_feasible_insert_position_test(instance, sequence_map, path_init_task_map, destroyed_task, robot_type):
#     feasible_position_set = get_all_position(sequence_map, path_init_task_map, robot_type)
#     to_delete_pos = []
#     for pos in feasible_position_set:
#         sequence_map_temp = copy_dict_int_dict(sequence_map)
#         path_init_task_map_temp = copy_dict_int_int(path_init_task_map)
#         insert_(sequence_map_temp, path_init_task_map_temp, destroyed_task, pos, robot_type)
#         fitness, feasible = cal_fitness(instance, sequence_map_temp, path_init_task_map_temp, destroyed_task)
#         if not feasible:
#             to_delete_pos.append(pos)
#     for pos in to_delete_pos:
#         feasible_position_set.discard(pos)
#
#     return feasible_position_set


def cal_fitness(instance, sequence_map, path_init_task_map):

    # if destroyed_task is not None:
    #     destroyed_task_required_robot = []
    #     for type in range(1, Config.ROBOT_TYPE_NUM + 1):
    #         if f"robot_{type}" in sequence_map[destroyed_task]:
    #             destroyed_task_required_robot.append(1)
    #         else:
    #             destroyed_task_required_robot.append(0)

    task_num = len(sequence_map.keys())

    info_map = {task: dict() for task in sequence_map.keys()}

    total_distance = 0
    total_tardiness = 0
    enabled_task_list = []
    task_require_robot_state = {task: [0 for _ in range(Config.ROBOT_TYPE_NUM)] for task in sequence_map.keys()}



    for path_index, init_task in path_init_task_map.items():
        if init_task != 0:
            robot_type = bisect.bisect_left(Config.INDEX_LIST, path_index) + 1
            task_require_robot_state[init_task][robot_type - 1] = 1
            info_map[init_task][f'robot_{robot_type}_pre_complete_time'] = 0
            if instance[init_task - 1][5] == task_require_robot_state[init_task]:
                enabled_task_list.append(init_task)
            # if destroyed_task is not None and init_task == destroyed_task and task_require_robot_state[init_task] == destroyed_task_required_robot and destroyed_task not in enabled_task_list:
            #     enabled_task_list.append(destroyed_task)


    completed_task_list = []

    while len(completed_task_list) < task_num:
        if len(enabled_task_list) == 0:
            return 1e6, False
        idx = random.randrange(len(enabled_task_list))
        enabled_task = enabled_task_list.pop(idx)

        source_position = [instance[enabled_task - 1][1], instance[enabled_task - 1][2]]

        arrival_time_map = {}
        robot_type_next_task_map = {}
        task_distance = 0
        for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
            if instance[enabled_task - 1][5][robot_type - 1] == 0:
                continue
            # if destroyed_task is not None and enabled_task == destroyed_task and destroyed_task_required_robot[robot_type - 1] == 0:
            #     continue
            pre_complete_time = info_map[enabled_task][f'robot_{robot_type}_pre_complete_time']
            if sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] == 0:
                pre_position = Config.DEPOT
            else:
                pre_position = [instance[sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] - 1][1],
                                instance[sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] - 1][2]]

            distance = get_distance(pre_position, source_position)
            travel_time = distance / Config.VELOCITY
            arrival_time = pre_complete_time + travel_time
            task_distance += distance

            arrival_time_map[robot_type] = arrival_time

            next_task = sequence_map[enabled_task][f'robot_{robot_type}_next_task']
            if next_task != 0:
                task_require_robot_state[next_task][robot_type - 1] = 1
                if task_require_robot_state[next_task] == instance[next_task - 1][5]:
                    enabled_task_list.append(next_task)
                # if destroyed_task is not None and next_task == destroyed_task and task_require_robot_state[
                #     next_task] == destroyed_task_required_robot and destroyed_task not in enabled_task_list:
                #     enabled_task_list.append(destroyed_task)
                robot_type_next_task_map[robot_type] = next_task

        execute_time = max(arrival_time_map.values())


        complete_time_list = []
        for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
            if instance[enabled_task - 1][5][robot_type - 1] == 0:
                continue
            complete_time = execute_time + instance[enabled_task - 1][4][robot_type - 1]
            next_task = sequence_map[enabled_task][f'robot_{robot_type}_next_task']
            if next_task != 0:
                info_map[next_task][f'robot_{robot_type}_pre_complete_time'] = complete_time

            complete_time_list.append(complete_time)

        final_complete_time = max(complete_time_list)

        # complete_time = execute_time + instance[enabled_task - 1][4]
        #
        # for robot_type, next_task in robot_type_next_task_map.items():
        #     info_map[next_task][f'robot_{robot_type}_pre_complete_time'] = complete_time

        task_tardiness = max(final_complete_time - instance[enabled_task - 1][3], 0)

        total_distance += task_distance
        total_tardiness += task_tardiness

        completed_task_list.append(enabled_task)

    fitness = total_distance * Config.WEIGHT + total_tardiness * (1 - Config.WEIGHT)

    return fitness, True



def get_animation_info(solution):
    # if destroyed_task is not None:
    #     destroyed_task_required_robot = []
    #     for type in range(1, Config.ROBOT_TYPE_NUM + 1):
    #         if f"robot_{type}" in sequence_map[destroyed_task]:
    #             destroyed_task_required_robot.append(1)
    #         else:
    #             destroyed_task_required_robot.append(0)

    instance = solution.instance
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()

    task_num = len(sequence_map.keys())

    info_map = {task: dict() for task in sequence_map.keys()}

    total_distance = 0
    total_tardiness = 0
    enabled_task_list = []
    task_require_robot_state = {task: [0 for _ in range(Config.ROBOT_TYPE_NUM)] for task in sequence_map.keys()}

    animation_info = {task: {} for task in range(1, len(instance) + 1)}

    for path_index, init_task in path_init_task_map.items():
        if init_task != 0:
            robot_type = bisect.bisect_left(Config.INDEX_LIST, path_index) + 1
            task_require_robot_state[init_task][robot_type - 1] = 1
            info_map[init_task][f'robot_{robot_type}_pre_complete_time'] = 0
            if instance[init_task - 1][5] == task_require_robot_state[init_task]:
                enabled_task_list.append(init_task)


    completed_task_list = []

    while len(completed_task_list) < task_num:
        if len(enabled_task_list) == 0:
            return 1e6, False
        idx = random.randrange(len(enabled_task_list))
        enabled_task = enabled_task_list.pop(idx)

        source_position = [instance[enabled_task - 1][1], instance[enabled_task - 1][2]]

        animation_info[enabled_task]["location"] = source_position

        arrival_time_map = {}
        robot_type_next_task_map = {}
        task_distance = 0
        for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
            if instance[enabled_task - 1][5][robot_type - 1] == 0:
                continue
            # if destroyed_task is not None and enabled_task == destroyed_task and destroyed_task_required_robot[robot_type - 1] == 0:
            #     continue
            pre_complete_time = info_map[enabled_task][f'robot_{robot_type}_pre_complete_time']
            if sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] == 0:
                pre_position = Config.DEPOT
            else:
                pre_position = [instance[sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] - 1][1],
                                instance[sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] - 1][2]]




            distance = get_distance(pre_position, source_position)
            travel_time = distance / Config.VELOCITY
            arrival_time = pre_complete_time + travel_time
            task_distance += distance

            arrival_time_map[robot_type] = arrival_time

            animation_info[enabled_task][f'robot_{robot_type}'] = sequence_map[enabled_task][f'robot_{robot_type}']
            animation_info[enabled_task][f'robot_{robot_type}_arrival_time'] = arrival_time

            next_task = sequence_map[enabled_task][f'robot_{robot_type}_next_task']
            if next_task != 0:
                task_require_robot_state[next_task][robot_type - 1] = 1
                if task_require_robot_state[next_task] == instance[next_task - 1][5]:
                    enabled_task_list.append(next_task)
                # if destroyed_task is not None and next_task == destroyed_task and task_require_robot_state[
                #     next_task] == destroyed_task_required_robot and destroyed_task not in enabled_task_list:
                #     enabled_task_list.append(destroyed_task)
                robot_type_next_task_map[robot_type] = next_task

        execute_time = max(arrival_time_map.values())

        animation_info[enabled_task]["start_time"] = execute_time


        complete_time_list = []
        for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
            if instance[enabled_task - 1][5][robot_type - 1] == 0:
                continue
            complete_time = execute_time + instance[enabled_task - 1][4][robot_type - 1]
            next_task = sequence_map[enabled_task][f'robot_{robot_type}_next_task']
            if next_task != 0:
                info_map[next_task][f'robot_{robot_type}_pre_complete_time'] = complete_time

            complete_time_list.append(complete_time)

            animation_info[enabled_task][f'robot_{robot_type}_complete_time'] = complete_time

        final_complete_time = max(complete_time_list)

        animation_info[enabled_task]["final_completion_time"] = final_complete_time
        animation_info[enabled_task]["deadline"] = instance[enabled_task - 1][3]

        # complete_time = execute_time + instance[enabled_task - 1][4]
        #
        # for robot_type, next_task in robot_type_next_task_map.items():
        #     info_map[next_task][f'robot_{robot_type}_pre_complete_time'] = complete_time

        task_tardiness = max(final_complete_time - instance[enabled_task - 1][3], 0)

        total_distance += task_distance
        total_tardiness += task_tardiness

        completed_task_list.append(enabled_task)

        animation_info[enabled_task]["tardiness"] = task_tardiness
        animation_info[enabled_task]["distance"] = task_distance

    fitness = total_distance * Config.WEIGHT + total_tardiness * (1 - Config.WEIGHT)

    return animation_info



