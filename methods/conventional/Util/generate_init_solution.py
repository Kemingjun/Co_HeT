import math
import random
from Util.util import *
from Util.Solution import Solution
from Util.load_data import read_excel
# from Util.operators import *

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
        # feasible_positions_test = get_feasible_insert_position_test(instance, destroyed_sequence_map, path_init_task_map, destroyed_task, current_robot_type)
        # if feasible_positions_test != feasible_positions:
        #     print("bug")
        for pos in feasible_positions:
            destroyed_sequence_map_temp = copy_dict_int_dict(destroyed_sequence_map)
            path_init_task_map_temp = copy_dict_int_int(path_init_task_map)
            insert_(destroyed_sequence_map_temp, path_init_task_map_temp, destroyed_task, pos, current_robot_type)

            search_insertions(instance, required_robot, current_robot_type + 1, destroyed_sequence_map_temp,
                              path_init_task_map_temp, destroyed_task, fitness_min_list, greedy_destroyed_sequence_map_list,
                              greedy_path_init_task_map_list)
#


def generate_solution_random(instance):
    task_num = len(instance)
    sequence_map = {}
    path_init_task_map = {robot: 0 for robot in range(1, Config.ROBOT_NUM + 1)}

    for task in range(1, task_num + 1):
        for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
            if instance[task - 1][5][robot_type - 1] == 1:
                feasible_positions = get_feasible_insert_position(sequence_map, path_init_task_map, task, robot_type)
                position = random.sample(feasible_positions, 1)[0]
                insert_(sequence_map, path_init_task_map, task, position, robot_type)
    solution = Solution(instance, sequence_map, path_init_task_map,)
    return solution


def generate_solution_greedy(instance):
    task_num = len(instance)
    task_list = [task for task in range(1, task_num + 1)]
    task_time_list = [[task, instance[task - 1][3]] for task in task_list]
    task_time_list = sorted(task_time_list, key=lambda x: x[1], reverse=False)
    destroyed_sequence_map = {}
    path_init_task_map = {robot: 0 for robot in range(1, Config.ROBOT_NUM + 1)}
    for task_time in task_time_list:
        destroyed_task = task_time[0]
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


def generate_solution_greedy_random_shuffle(instance):
    task_num = len(instance)
    task_list = [task for task in range(1, task_num + 1)]
    random.shuffle(task_list)
    # task_time_list = [[task, instance[task - 1][3]] for task in task_list]
    # task_time_list = sorted(task_time_list, key=lambda x: x[1], reverse=False)
    destroyed_sequence_map = {}
    path_init_task_map = {robot: 0 for robot in range(1, Config.ROBOT_NUM + 1)}
    for task in task_list:
        greedy_destroyed_sequence_map_list = [None]
        greedy_path_init_task_map_list = [None]
        fitness_min_list = [1e9]

        required_robot = instance[task - 1][5]
        search_insertions(
            instance=instance,
            required_robot=required_robot,
            current_robot_type=1,
            destroyed_sequence_map=destroyed_sequence_map,
            path_init_task_map=path_init_task_map,
            destroyed_task=task,
            fitness_min_list=fitness_min_list,
            greedy_destroyed_sequence_map_list=greedy_destroyed_sequence_map_list,
            greedy_path_init_task_map_list=greedy_path_init_task_map_list,
        )

        if greedy_destroyed_sequence_map_list[0] is None:
            print(f"BUG: Task {task} has no feasible insertion!")

        destroyed_sequence_map = greedy_destroyed_sequence_map_list[0]
        path_init_task_map = greedy_path_init_task_map_list[0]

    return Solution(instance, destroyed_sequence_map, path_init_task_map)



def generate_solution_nearest(instance, factor=0.9):
    task_num = len(instance)
    task_list = list(range(1, task_num + 1))

    robot_state_map = {i: [0, [0, 0]] for i in range(1, Config.ROBOT_NUM + 1)}

    path_map = {i: [] for i in range(1, Config.ROBOT_NUM + 1)}

    while len(task_list) != 0:
        robot_list = []
        for type in range(1, Config.ROBOT_TYPE_NUM + 1):
            type_agent_state_map = {k: v for k, v in robot_state_map.items() if Config.TYPE_LIST[type - 1][0] <= k < Config.TYPE_LIST[type - 1][1]}
            min_type_robot = min(type_agent_state_map, key=lambda k: type_agent_state_map[k][0])
            robot_list.append(min_type_robot)
        task_cost_list = []
        for task in task_list:
            task_deadline = instance[task - 1][3]
            task_position_x = instance[task - 1][1]
            task_position_y = instance[task - 1][2]
            cost = 0
            for robot in robot_list:
                robot_time = robot_state_map[robot][0]
                robot_position_x = robot_state_map[robot][1][0]
                robot_position_y = robot_state_map[robot][1][1]
                distance = math.fabs(task_position_x - robot_position_x) + math.fabs(task_position_y - robot_position_y)
                delta_time = task_deadline - robot_time
                cost += (factor * distance + (1 - factor) * delta_time)
            task_cost_list.append([task, cost])
        task_cost_list.sort(key=lambda x: x[1])
        selected_task = task_cost_list[0][0]

        arrival_time_list = []
        for robot in robot_list:
            robot_time = robot_state_map[robot][0]
            robot_position_x = robot_state_map[robot][1][0]
            robot_position_y = robot_state_map[robot][1][1]
            distance = math.fabs(robot_position_x - instance[selected_task - 1][1]) + math.fabs(robot_position_y - instance[selected_task - 1][2])
            arrival_time = robot_time + distance / Config.VELOCITY
            arrival_time_list.append(arrival_time)

        start_time = max(arrival_time_list)

        for type, robot in enumerate(robot_list):
            path_map[robot].append(selected_task)
            robot_state_map[robot][0] = start_time + instance[4][type]
            robot_state_map[robot][1] = [instance[selected_task - 1][1], instance[selected_task - 1][2]]
        task_list.remove(selected_task)

    sequence_map, path_init_task_map = path_map2sequence_map(path_map)
    solution = Solution(instance, sequence_map, path_init_task_map)
    return solution

# if __name__ == "__main__":
#     instance = read_excel("N100_K2_M12_I3.xlsx")
#     for i in range(1, 10):
#         solution = generate_solution_nearest(instance, 0.1 + i * 0.1)
#         print(f"fitness:{solution.get_fitness()} factor: {0.1 + i * 0.1}")
#     pass


