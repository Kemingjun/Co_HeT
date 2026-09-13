from Util.generate_init_solution import generate_solution_nearest
from Util.load_data import read_excel
from Util.operators import *
import numpy as np
import math
# from Util.ALNS_config import ALNSConfig
import time

d_num_coefficient = 0.2  # destroy涓暟鐨勬瘮渚?
T_coefficient = 0.1  # removal proportion
rho = 0.1  # reaction factor
l_s = 10  # update interval
sigma_1 = 33  # score 1
sigma_2 = 13  # score 2
sigma_3 = 9  # score 3

C = 5  # computation time

destructOperatorList = [destroy_random,
                        destroy_worst_cost,
                        destroy_worst_distance,
                        destroy_worst_tardiness]

constructOperatorList = [repair_greedy,
                         repair_greedy_urgency,
                         repair_greedy_cost]


d_operator_num = len(destructOperatorList)  # 鐮村潖绠楀瓙涓暟
c_operator_num = len(constructOperatorList)  # 閲嶅缓绠楀瓙涓暟


def destruct_construct(current_solution, d_num, wDestruct, wConstruct):
    destruct_index = np.random.choice(np.arange(len(wDestruct)), p=np.array(wDestruct) / sum(wDestruct))
    construct_index = np.random.choice(np.arange(len(wConstruct)), p=np.array(wConstruct) / sum(wConstruct))
    destroyed_info = destructOperatorList[destruct_index](current_solution, d_num)
    new_solution = constructOperatorList[construct_index](*destroyed_info, current_solution)
    return new_solution, destruct_index, construct_index



def ALNS(instance_name, max_iterations=100, time_limit=3600, seed=None, verbose=True):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    instance = read_excel(instance_name + ".xlsx")
    timesDestruct = [0 for _ in range(d_operator_num)]
    timesConstruct = [0 for _ in range(c_operator_num)]
    totalScoreDestruct = [0 for _ in range(d_operator_num)]
    totalScoreConstruct = [0 for _ in range(c_operator_num)]
    wDestruct = [1 for _ in range(d_operator_num)]
    wConstruct = [1 for _ in range(c_operator_num)]

    solution_table = {}
    task_num = len(instance)
    d_num = math.ceil(task_num * d_num_coefficient)

    init_start = time.time()
    solution = generate_solution_nearest(instance)
    init_end = time.time()
    if verbose:
        print(f"Initial solution generation time: {init_end - init_start}")

    solution_table[solution.hash_key] = solution
    current_fitness = solution.get_fitness()
    if verbose:
        print(f"Initial fitness: {current_fitness}")

    best_solution = solution
    best_fitness = current_fitness
    count = 0
    constant_t = T_coefficient
    if verbose:
        print(f"Temperature: {constant_t}")
        duration = task_num ** 2 * sum(Config.ROBOT_NUM_LIST) * C / 1000
        print(f"Estimated duration indicator: {duration}")

    start_t = time.time()
    while count < max_iterations and time.time() - start_t < time_limit:
        count += 1
        new_solution, destruct_index, construct_index = destruct_construct(
            solution, d_num, wDestruct, wConstruct
        )
        is_accept = False
        is_new = False

        if new_solution.hash_key not in solution_table.keys():
            solution_table[new_solution.hash_key] = new_solution
            is_new = True

        new_fitness = new_solution.get_fitness()
        scoreDestruct = 0
        scoreConstruct = 0

        if new_fitness < current_fitness:
            is_accept = True
            p_a = 1.0
            solution = new_solution
            current_fitness = new_fitness
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                scoreDestruct = sigma_1
                scoreConstruct = sigma_1
            elif is_new:
                scoreDestruct = sigma_2
                scoreConstruct = sigma_2
        elif new_fitness == current_fitness:
            is_accept = True
            if is_new:
                scoreDestruct = sigma_2
                scoreConstruct = sigma_2
            p_a = 0
        else:
            p_a = math.exp((current_fitness - new_fitness) / constant_t)
            if random.random() < p_a:
                is_accept = True
                solution = new_solution
                current_fitness = new_fitness
                if is_new:
                    scoreDestruct = sigma_3
                    scoreConstruct = sigma_3

        timesDestruct[destruct_index] += 1
        timesConstruct[construct_index] += 1
        totalScoreDestruct[destruct_index] += scoreDestruct
        totalScoreConstruct[construct_index] += scoreConstruct

        if count % l_s == 0:
            for i in range(d_operator_num):
                dTime = timesDestruct[i] if timesDestruct[i] != 0 else 1
                wDestruct[i] = wDestruct[i] * (1 - rho) + rho * totalScoreDestruct[i] / dTime
                totalScoreDestruct[i] = 0
                timesDestruct[i] = 0
            for i in range(c_operator_num):
                cTime = timesConstruct[i] if timesConstruct[i] != 0 else 1
                wConstruct[i] = wConstruct[i] * (1 - rho) + rho * totalScoreConstruct[i] / cTime
                totalScoreConstruct[i] = 0
                timesConstruct[i] = 0

        if verbose:
            print(
                f"Iteration {count}, best fitness: {round(best_fitness, 4)}, "
                f"current fitness: {round(current_fitness, 4)}, new fitness: {round(new_fitness, 4)}, "
                f"d_index: {destruct_index}, c_index: {construct_index}, "
                f"p: {round(p_a, 3)}, accepted: {is_accept}, time: {round(time.time() - start_t, 4)}"
            )

    if verbose:
        print(f"best fitness: {best_fitness} best solution: {best_solution}")
    return best_solution
