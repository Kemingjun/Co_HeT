from Util.Config import Config
import bisect
import random
import math
import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, filename=LOG_DIR / "info.log")
def copy_set_int(original_set):
    """
    对set进行深拷贝
    set中元素的类型为int
    :param original_set:
    :return:
    """
    copied_set = set()
    for item in original_set:
        copied_set.add(item)

    return copied_set


def copy_list_int(original_list):
    """
    对set进行深拷贝
    set中元素的类型为int
    :param original_list:
    :return:
    """
    copied_list = []
    for item in copied_list:
        copied_list.append(item)

    return copied_list


def copy_dict_int_int(original_dict):
    # 创建一个新的空字典
    copied_dict = {}

    # 遍历原字典中的每个键值对，将其添加到新的字典中
    for key, value in original_dict.items():
        copied_dict[key] = value

    return copied_dict


def copy_dict_int_list(original_dict):
    # 创建一个新的空字典
    copied_dict = {}

    # 遍历原字典中的每个键值对，将其添加到新的字典中
    for key, value in original_dict.items():
        copied_list = []
        for v in value:
            copied_list.append(v)
        copied_dict[key] = copied_list

    return copied_dict


def copy_dict_int_dict(original_dict):
    # 创建一个新的空字典
    copied_dict = {}

    # 遍历原字典中的每个键值对
    for key, inner_dict in original_dict.items():
        # 创建一个新的内层字典
        copied_inner_dict = {}

        # 遍历内层字典中的每个键值对
        for inner_key, value in inner_dict.items():
            copied_inner_dict[inner_key] = value

        # 将拷贝后的内层字典添加到外层字典中
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
    # 每个任务的前序和后续信息
    """
    每个任务包括，每种类型的robot的指定执行的robot，以及该robot的前继后继任务
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
    distance = abs(source_position[0] - destination_position[0]) + abs(source_position[1] - destination_position[1])
    # distance = math.hypot(source_position[0] - destination_position[0], source_position[1] - destination_position[1])
    return distance


def task_id(task_info):
    return int(task_info[0])


def supply_pos(task_info):
    return [task_info[1], task_info[2]]


def handover_pos(task_info):
    return [task_info[3], task_info[4]]


def delivery_pos(task_info):
    return [task_info[5], task_info[6]]


def deadline(task_info):
    return task_info[7]


def required_robot(task_info):
    return task_info[8]


def _pre_task(sequence_map, task, robot_type):
    return sequence_map[task][f'robot_{robot_type}_pre_task']


def _next_task(sequence_map, task, robot_type):
    return sequence_map[task][f'robot_{robot_type}_next_task']


def _task_delivery_or_depot(instance, task):
    if task == 0:
        return Config.DEPOT
    return delivery_pos(instance[task - 1])


def _task_handover_or_depot(instance, task):
    if task == 0:
        return Config.DEPOT
    return handover_pos(instance[task - 1])


def evaluate_real_world_schedule(instance, sequence_map, path_init_task_map, info_map=None):
    task_num = len(sequence_map.keys())
    if info_map is None:
        info_map = {task: {} for task in sequence_map.keys()}

    total_distance = 0
    total_travel_time = 0
    total_tardiness = 0
    enabled_task_list = []
    task_require_robot_state = {task: [0 for _ in range(Config.ROBOT_TYPE_NUM)] for task in sequence_map.keys()}

    for path_index, init_task in path_init_task_map.items():
        if init_task != 0:
            robot_type = bisect.bisect_left(Config.INDEX_LIST, path_index) + 1
            task_require_robot_state[init_task][robot_type - 1] = 1
            info_map[init_task][f'robot_{robot_type}_pre_complete_time'] = 0
            if required_robot(instance[init_task - 1]) == task_require_robot_state[init_task]:
                enabled_task_list.append(init_task)

    completed_task_list = []
    while len(completed_task_list) < task_num:
        if len(enabled_task_list) == 0:
            return 1e6, total_distance, total_travel_time, total_tardiness, False, info_map

        enabled_task = enabled_task_list.pop(0)
        task_info = instance[enabled_task - 1]

        carrier_pre_task = _pre_task(sequence_map, enabled_task, Config.CARRIER_TYPE)
        shuttle_pre_task = _pre_task(sequence_map, enabled_task, Config.SHUTTLE_TYPE)
        forklift_pre_task = _pre_task(sequence_map, enabled_task, Config.FORKLIFT_TYPE)

        carrier_pre_complete = info_map[enabled_task][f'robot_{Config.CARRIER_TYPE}_pre_complete_time']
        shuttle_pre_complete = info_map[enabled_task][f'robot_{Config.SHUTTLE_TYPE}_pre_complete_time']
        forklift_pre_complete = info_map[enabled_task][f'robot_{Config.FORKLIFT_TYPE}_pre_complete_time']

        carrier_pre_position = _task_delivery_or_depot(instance, carrier_pre_task)
        shuttle_ad_position = _task_delivery_or_depot(instance, shuttle_pre_task)
        forklift_pre_position = _task_handover_or_depot(instance, forklift_pre_task)

        supply_position = supply_pos(task_info)
        handover_position = handover_pos(task_info)
        delivery_position = delivery_pos(task_info)

        carrier_to_shuttle_distance = get_distance(carrier_pre_position, shuttle_ad_position)
        shuttle_to_handover_distance = get_distance(shuttle_ad_position, handover_position)
        handover_to_delivery_distance = get_distance(handover_position, delivery_position)
        forklift_to_supply_distance = get_distance(forklift_pre_position, supply_position)
        supply_to_handover_distance = get_distance(supply_position, handover_position)

        carrier_arrive_ad = carrier_pre_complete + carrier_to_shuttle_distance / Config.CARRIER_VELOCITY
        attach_start = max(carrier_arrive_ad, shuttle_pre_complete)
        attach_end = attach_start + Config.CARRIER_SHUTTLE_COUPLING_TIME
        carrier_shuttle_arrive_handover = attach_end + shuttle_to_handover_distance / Config.CARRIER_VELOCITY

        forklift_arrive_supply = forklift_pre_complete + forklift_to_supply_distance / Config.FORKLIFT_VELOCITY
        forklift_depart_supply = forklift_arrive_supply + Config.FORKLIFT_PICKUP_TIME
        forklift_arrive_handover = forklift_depart_supply + supply_to_handover_distance / Config.FORKLIFT_VELOCITY

        handover_start = max(carrier_shuttle_arrive_handover, forklift_arrive_handover)
        handover_end = handover_start + Config.SOURCE_HANDOVER_TIME
        forklift_release = handover_end

        arrive_destination = handover_end + handover_to_delivery_distance / Config.CARRIER_VELOCITY
        carrier_release = arrive_destination + Config.CARRIER_SHUTTLE_DECOUPLING_TIME
        shuttle_release = (
            carrier_release
            + Config.SHUTTLE_UNLOADING_TIME
            + Config.DELIVERY_STATION_PROCESSING_TIME
        )

        carrier_distance = (
            carrier_to_shuttle_distance
            + shuttle_to_handover_distance
            + handover_to_delivery_distance
        )
        forklift_distance = forklift_to_supply_distance + supply_to_handover_distance
        task_distance = carrier_distance + forklift_distance
        task_travel_time = (
            carrier_distance / Config.CARRIER_VELOCITY
            + forklift_distance / Config.FORKLIFT_VELOCITY
        )
        task_tardiness = max(shuttle_release - deadline(task_info), 0)
        task_cost = task_travel_time * Config.WEIGHT + task_tardiness * (1 - Config.WEIGHT)

        info_map[enabled_task].update({
            'carrier_arrive_ad_time': carrier_arrive_ad,
            'attach_start_time': attach_start,
            'attach_end_time': attach_end,
            'carrier_shuttle_arrive_handover_time': carrier_shuttle_arrive_handover,
            'forklift_arrive_supply_time': forklift_arrive_supply,
            'forklift_depart_supply_time': forklift_depart_supply,
            'forklift_arrive_handover_time': forklift_arrive_handover,
            'handover_start_time': handover_start,
            'handover_end_time': handover_end,
            'forklift_release_time': forklift_release,
            'arrive_destination_time': arrive_destination,
            'carrier_release_time': carrier_release,
            'shuttle_release_time': shuttle_release,
            'complete_time': shuttle_release,
            'carrier_distance': carrier_distance,
            'forklift_distance': forklift_distance,
            'distance': task_distance,
            'travel_time': task_travel_time,
            'tardiness': task_tardiness,
            'cost': task_cost,
        })

        next_times = {
            Config.CARRIER_TYPE: carrier_release,
            Config.SHUTTLE_TYPE: shuttle_release,
            Config.FORKLIFT_TYPE: forklift_release,
        }
        for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
            next_task = _next_task(sequence_map, enabled_task, robot_type)
            if next_task != 0:
                info_map[next_task][f'robot_{robot_type}_pre_complete_time'] = next_times[robot_type]
                task_require_robot_state[next_task][robot_type - 1] = 1
                if task_require_robot_state[next_task] == required_robot(instance[next_task - 1]):
                    enabled_task_list.append(next_task)

        total_distance += task_distance
        total_travel_time += task_travel_time
        total_tardiness += task_tardiness
        completed_task_list.append(enabled_task)

    fitness = total_travel_time * Config.WEIGHT + total_tardiness * (1 - Config.WEIGHT)
    return fitness, total_distance, total_travel_time, total_tardiness, True, info_map


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
        # 插入的是一个非空的path
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

    position_set = {(task, 0) for task in tasks_with_robot_type}  # 0表示该task之前
    position_set = position_set.union({(task, 1) for task in tasks_with_robot_type})  # 1表示该task之后
    '''
        以下执行去重操作
        [1,2,3]
        (2,1) 和 (3, 0)位置等价
        此处默认去除(2, 1)
    '''
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
                    # 该任务已经分配了该类型的robot
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
                        # 在robot_type那条链上的下下个任务
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
    fitness, _, _, _, feasible, _ = evaluate_real_world_schedule(
        instance,
        sequence_map,
        path_init_task_map,
    )
    return fitness, feasible
