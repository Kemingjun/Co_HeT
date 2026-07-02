from Util.load_data import read_excel
from Util.generate_init_solution import generate_solution_random, generate_solution_greedy
import time
from Util.Config import Config
from Util.ALNS_config import ALNSConfig
import numpy as np
import random
from Util.util import *
from Util.Solution import Solution

POP_SIZE = 100
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1



def crossover_and_mutation(pop):
    new_pop = []
    for father in pop:
        child = father
        if np.random.rand() < CROSSOVER_RATE:
            mother = random.choice(pop)
            child = cross_over_solution(father, mother)
        child = mutation(child)
        new_pop.append(child)
    return new_pop


def mutation(solution):
    if np.random.rand() > MUTATION_RATE:
        return solution
    task = random.randint(1, solution.task_num)

    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()

    remove_(sequence_map, path_init_task_map, task)

    for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
        feasible_position_set = get_feasible_insert_position(sequence_map, path_init_task_map, task, robot_type)
        position = random.sample(feasible_position_set, 1)[0]
        insert_(sequence_map, path_init_task_map, task, position, robot_type)
    return Solution(solution.instance, sequence_map, path_init_task_map)



def cross_over_solution(solution_father, solution_mother):
    task_num = solution_father.task_num
    task_list = list(range(1, task_num + 1))
    random.shuffle(task_list)
    cross_task_num = 0
    father_path_map = solution_father.get_path_map()
    mother_path_map = solution_mother.get_path_map()
    for task in task_list:
        father_path_map_temp = copy_dict_int_list(father_path_map)
        mother_path_map_temp = copy_dict_int_list(mother_path_map)
        crossed_path_map = cross_over_task(father_path_map_temp, mother_path_map_temp, task)
        crossed_sequence_map, crossed_path_init_task_map = path_map2sequence_map(crossed_path_map)
        fitness, feasible = cal_fitness_feasible(solution_mother.instance, crossed_sequence_map, crossed_path_init_task_map)
        if feasible:
            cross_task_num += 1
            mother_path_map = crossed_path_map
        if cross_task_num >= 1:
            break
    crossed_sequence_map, crossed_path_init_task_map = path_map2sequence_map(mother_path_map)
    return Solution(solution_father.instance, crossed_sequence_map, crossed_path_init_task_map)

# def get_type(x):
#     for idx, (low, high) in enumerate(Config.TYPE_LIST, start=1):
#         if low <= x < high:
#             return idx
def get_position(path_map, task):
    position_list = []
    for path_index, path in path_map.items():
        for ind, _task in enumerate(path):
            if _task == task:
                position_list.append((path_index, ind))
    return position_list



def cross_over_task(father_path_map_temp, mother_path_map_temp, task):
    father_position_list = get_position(father_path_map_temp, task)
    mother_position_list = get_position(mother_path_map_temp, task)

    for mother_position in mother_position_list:
        mother_path_map_temp[mother_position[0]].pop(mother_position[1])

    for father_position in father_position_list:
        mother_path_map_temp[father_position[0]].insert(father_position[1], task)
    
    return mother_path_map_temp


def select(pop):
    fitness_list = np.array([s.get_fitness() for s in pop])
    # hash_key_list = np.array([hash_key for hash_key in pop_map.keys()])
    # fitness_list = np.array([pop_map[hash_key].get_fitness() for hash_key in hash_key_list])
    for j in range(len(fitness_list)):
        fitness_list[j] = - fitness_list[j]

    fitness_list = (fitness_list - np.min(fitness_list)) + 1e-3
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness_list) / (fitness_list.sum()))
    return np.array(pop)[idx]



def cal_fitness_feasible(instance, sequence_map, path_init_task_map):


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

def GA(instance_name):
    instance = read_excel(instance_name + ".xlsx")
    task_num = len(instance)
    pop = []
    best_solution = generate_solution_greedy(instance)
    best_fitness = best_solution.get_fitness()
    pop.append(best_solution)

    for _ in range(1, int(POP_SIZE)):
        pop.append(generate_solution_random(instance))

    start_t = time.time()
    count = 0

    duration = task_num ** 2 * sum(Config.ROBOT_NUM_LIST) * ALNSConfig.C / 1000

    while time.time() - start_t <= 100:
        count += 1
        for solution in pop:
            fitness = solution.get_fitness()
            if fitness < best_fitness:
                best_solution = solution
                best_fitness = fitness

        new_pop = crossover_and_mutation(pop)
        pop = select(new_pop)

    return best_fitness



if __name__ == "__main__":

    # solution = generate_solution_nearest2(instance)
    # fitness = solution.get_fitness()
    # best_sequence_map, best_path_init_task_map, best_fitness = opt_2_reverse_neighbor_best_solution(solution)
    # init_optimal_solution()
    instance_name = "N20_K2_M12_I1"
    # instance = read_excel(instance_name + ".xlsx")
    GA(instance_name)
    pass


