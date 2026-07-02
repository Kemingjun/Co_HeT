
from Util.load_data import read_excel
import math
from Util.generate_init_solution import generate_solution_random, generate_solution_nearest
import time
from Util.operators import destroy_random, repair_greedy
import random
from Util.Solution import Solution
from Util.util import *

d_num_coefficient = 0.2
T_coefficient = 0.1

def destruct_construct(current_solution, d_num):
    destroyed_info = destroy_random(current_solution, d_num)
    new_solution = repair_greedy(*destroyed_info, current_solution)
    return new_solution

def local_search(solution, start_t, duration):

    task_list = list(range(1, solution.task_num + 1))
    current_fitness = solution.get_fitness()
    current_solution = solution
    while True:
        random.shuffle(task_list)
        flag = False
        for task in task_list:
            sequence_map = current_solution.get_sequence_map()
            path_init_task_map = current_solution.get_path_init_task_map()
            if time.time() - start_t > duration:
                return Solution(solution.instance, sequence_map, path_init_task_map)
            remove_(sequence_map, path_init_task_map, task)
            new_solution = repair_greedy(sequence_map, path_init_task_map, [task], solution)

            if new_solution.get_fitness() < current_fitness:
                current_solution = new_solution
                current_fitness = new_solution.get_fitness()
                flag = True
                break
        if not flag:
            break
    return current_solution


def local_search_type2(solution, start_t, duration):
    """
    
    :param solution:
    :param start_t:
    :param duration:
    :return:
    """
    task_list = list(range(1, solution.task_num + 1))
    current_fitness = solution.get_fitness()
    current_solution = solution
    random.shuffle(task_list)
    # while True:
    #     random.shuffle(task_list)
    #     flag = False
    for task in task_list:
        sequence_map = current_solution.get_sequence_map()
        path_init_task_map = current_solution.get_path_init_task_map()
        if time.time() - start_t > duration:
            return Solution(solution.instance, sequence_map, path_init_task_map)
        remove_(sequence_map, path_init_task_map, task)
        new_solution = repair_greedy(sequence_map, path_init_task_map, [task], solution)

        if new_solution.get_fitness() < current_fitness:
            current_solution = new_solution
            current_fitness = new_solution.get_fitness()
            break
                # flag = True
                # break
        # if not flag:
        #     break
    if current_solution.get_fitness() < solution.get_fitness():
        print("success")
    return current_solution






def IGA(instance_name):
    instance = read_excel(instance_name + ".xlsx")
    task_num = len(instance)
    d_num = math.ceil(task_num * d_num_coefficient)
    init_start = time.time()
    solution = generate_solution_nearest(instance)
    init_end = time.time()

    current_fitness = solution.get_fitness()

    best_solution = solution
    best_fitness = current_fitness

    start_t = time.time()

    count = 0

    duration = 1800

    while count <= 100:
    # while time.time() - start_t <= duration:
        count += 1
        neighbor_solution = local_search_type2(solution, start_t, duration)
        new_solution = destruct_construct(neighbor_solution, d_num)


        new_fitness = new_solution.get_fitness()
        if new_fitness < current_fitness:
            solution = new_solution
            current_fitness = new_fitness
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
        elif new_fitness == current_fitness:
            pass
        else:
            p_a = math.exp((current_fitness - new_fitness) / T_coefficient)
            if random.random() < p_a:
                solution = new_solution
                current_fitness = new_fitness


    return best_fitness

if __name__ == "__main__":
    # init_optimal_solution()
    instance_name = "N20_K2_M12_I1"
    # instance = read_excel(instance_name + ".xlsx")
    # T = get_T(instance)
    IGA(instance_name)
    # update_optimal_solution()
    pass





