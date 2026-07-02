import random

from Util.util import *
from Util.Solution import Solution
from Util.load_data import read_excel
from Util.generate_init_solution import generate_solution_random


def destroy_random(solution, d_num):
    path_init_task_map = solution.get_path_init_task_map()
    destroyed_sequence_map = solution.get_sequence_map()
    destroyed_task_list = random.sample([task for task in range(1, solution.task_num + 1)], d_num)

    for destroyed_task in destroyed_task_list:
        remove_(destroyed_sequence_map, path_init_task_map, destroyed_task)

    return destroyed_sequence_map, path_init_task_map, destroyed_task_list


def destroy_worst_cost(solution, d_num):
    task_cost_list = []
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()
    destroyed_task_list = []
    task_info_map = solution.info_map
    task_list = list(sequence_map.keys())

    for task in task_list:
        cost = task_info_map[task]['distance']
        task_cost_list.append([task, cost])
    task_cost_list = sorted(task_cost_list, key=lambda x: x[1], reverse=True)
    for i in range(d_num):
        destroyed_task_list.append(task_cost_list[i][0])

    for task in destroyed_task_list:
        remove_(sequence_map, path_init_task_map, task)
    return sequence_map, path_init_task_map, destroyed_task_list


def destroy_worst_distance(solution, d_num):
    """
    distancedistance
    :param d_num:
    :param solution:
    :return:
    """
    task_distance_list = []
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()
    destroyed_task_list = []
    task_list = list(sequence_map.keys())
    task_info_map = solution.info_map
    if task_info_map is None:
        solution.get_fitness()
        task_info_map = solution.info_map

    for task in task_list:
        distance = task_info_map[task]['distance']
        task_distance_list.append([task, distance])

    task_distance_list = sorted(task_distance_list, key=lambda x: x[1], reverse=True)
    for i in range(d_num):
        destroyed_task_list.append(task_distance_list[i][0])

    for task in destroyed_task_list:
        remove_(sequence_map, path_init_task_map, task)
    return sequence_map, path_init_task_map, destroyed_task_list


def destroy_worst_tardiness(solution, d_num):
    """
    distance
    :param d_num:
    :param solution:
    :return:
    """
    task_tardiness_list = []
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()
    destroyed_task_list = []
    task_list = list(sequence_map.keys())
    task_info_map = solution.info_map

    for task in task_list:
        distance = task_info_map[task]['tardiness']
        task_tardiness_list.append([task, distance])

    task_tardiness_list = sorted(task_tardiness_list, key=lambda x: x[1], reverse=True)
    for i in range(d_num):
        destroyed_task_list.append(task_tardiness_list[i][0])

    for task in destroyed_task_list:
        remove_(sequence_map, path_init_task_map, task)
    return sequence_map, path_init_task_map, destroyed_task_list





def repair_greedy(destroyed_sequence_map, path_init_task_map, destroyed_task_list, current_solution):
    instance = current_solution.instance
    random.shuffle(destroyed_task_list)
    for destroyed_task in destroyed_task_list:
        greedy_destroyed_sequence_map_list = [None]
        greedy_path_init_task_map_list = [None]
        fitness_min_list = [1e9]

        required_robot = instance[destroyed_task - 1][5]
        search_insertions(
            instance=instance,
            required_robot=required_robot,
            current_robot_type=1,
            destroyed_sequence_map=destroyed_sequence_map,
            path_init_task_map=path_init_task_map,
            destroyed_task=destroyed_task,
            fitness_min_list=fitness_min_list,
            greedy_destroyed_sequence_map_list=greedy_destroyed_sequence_map_list,
            greedy_path_init_task_map_list=greedy_path_init_task_map_list,
        )

        if greedy_destroyed_sequence_map_list[0] is None:
            print(f"BUG: Task {destroyed_task} has no feasible insertion!")

        destroyed_sequence_map = greedy_destroyed_sequence_map_list[0]
        path_init_task_map = greedy_path_init_task_map_list[0]

    return Solution(instance, destroyed_sequence_map, path_init_task_map)

def repair_greedy_urgency(destroyed_sequence_map, path_init_task_map, destroyed_task_list, current_solution):
    instance = current_solution.instance
    # random.shuffle(destroyed_task_list)
    destroyed_task_l_time = []
    for task in destroyed_task_list:
        destroyed_task_l_time.append([task, instance[task - 1][3]])
    destroyed_task_l_time = sorted(destroyed_task_l_time, key=lambda x: x[1], reverse=False)
    for destroyed_task_ in destroyed_task_l_time:
        destroyed_task = destroyed_task_[0]
        greedy_destroyed_sequence_map_list = [None]
        greedy_path_init_task_map_list = [None]
        fitness_min_list = [1e9]

        required_robot = instance[destroyed_task - 1][5]
        search_insertions(
            instance=instance,
            required_robot=required_robot,
            current_robot_type=1,
            destroyed_sequence_map=destroyed_sequence_map,
            path_init_task_map=path_init_task_map,
            destroyed_task=destroyed_task,
            fitness_min_list=fitness_min_list,
            greedy_destroyed_sequence_map_list=greedy_destroyed_sequence_map_list,
            greedy_path_init_task_map_list=greedy_path_init_task_map_list,
        )

        if greedy_destroyed_sequence_map_list[0] is None:
            print(f"BUG: Task {destroyed_task} has no feasible insertion!")

        destroyed_sequence_map = greedy_destroyed_sequence_map_list[0]
        path_init_task_map = greedy_path_init_task_map_list[0]

    return Solution(instance, destroyed_sequence_map, path_init_task_map)


def repair_greedy_cost(destroyed_sequence_map, path_init_task_map, destroyed_task_list, current_solution):
    info_map = current_solution.info_map
    instance = current_solution.instance
    # random.shuffle(destroyed_task_list)
    destroyed_task_cost_list = []
    for task in destroyed_task_list:
        destroyed_task_cost_list.append([task, info_map[task]['cost']])
    destroyed_task_cost_list = sorted(destroyed_task_cost_list, key=lambda x: x[1], reverse=True)
    for destroyed_task_ in destroyed_task_cost_list:
        destroyed_task = destroyed_task_[0]
        greedy_destroyed_sequence_map_list = [None]
        greedy_path_init_task_map_list = [None]
        fitness_min_list = [1e9]

        required_robot = instance[destroyed_task - 1][5]
        search_insertions(
            instance=instance,
            required_robot=required_robot,
            current_robot_type=1,
            destroyed_sequence_map=destroyed_sequence_map,
            path_init_task_map=path_init_task_map,
            destroyed_task=destroyed_task,
            fitness_min_list=fitness_min_list,
            greedy_destroyed_sequence_map_list=greedy_destroyed_sequence_map_list,
            greedy_path_init_task_map_list=greedy_path_init_task_map_list,
        )

        if greedy_destroyed_sequence_map_list[0] is None:
            print(f"BUG: Task {destroyed_task} has no feasible insertion!")

        destroyed_sequence_map = greedy_destroyed_sequence_map_list[0]
        path_init_task_map = greedy_path_init_task_map_list[0]

    return Solution(instance, destroyed_sequence_map, path_init_task_map)



def search_insertions(instance, required_robot, current_robot_type, destroyed_sequence_map, path_init_task_map,
                      destroyed_task, fitness_min_list, greedy_destroyed_sequence_map_list, greedy_path_init_task_map_list):
    if current_robot_type == len(required_robot) + 1:
        fitness, _ = cal_fitness(instance, destroyed_sequence_map, path_init_task_map)
        if fitness < fitness_min_list[0]:
            fitness_min_list[0] = fitness
            greedy_destroyed_sequence_map_list[0] = copy_dict_int_dict(destroyed_sequence_map)
            greedy_path_init_task_map_list[0] = copy_dict_int_int(path_init_task_map)
        return
    if required_robot[current_robot_type - 1] == 0:
        search_insertions(instance, required_robot, current_robot_type + 1, destroyed_sequence_map, path_init_task_map,
                      destroyed_task, fitness_min_list, greedy_destroyed_sequence_map_list, greedy_path_init_task_map_list)
    else:
        feasible_positions = get_feasible_insert_position(destroyed_sequence_map, path_init_task_map, destroyed_task, current_robot_type)
        for pos in feasible_positions:
            destroyed_sequence_map_temp = copy_dict_int_dict(destroyed_sequence_map)
            path_init_task_map_temp = copy_dict_int_int(path_init_task_map)
            insert_(destroyed_sequence_map_temp, path_init_task_map_temp, destroyed_task, pos, current_robot_type)

            search_insertions(instance, required_robot, current_robot_type + 1, destroyed_sequence_map_temp,
                              path_init_task_map_temp, destroyed_task, fitness_min_list, greedy_destroyed_sequence_map_list,
                              greedy_path_init_task_map_list)



