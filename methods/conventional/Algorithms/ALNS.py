from Util.generate_init_solution import generate_solution_nearest
from Util.load_data import read_excel
from Util.operators import *
import numpy as np
import math
# from Util.ALNS_config import ALNSConfig
import time
# import Util.result_record as RR

d_num_coefficient = 0.2  # destroy个数的比例
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


d_operator_num = len(destructOperatorList)  # 破坏算子个数
c_operator_num = len(constructOperatorList)  # 重建算子个数


def destruct_construct(current_solution, d_num, wDestruct, wConstruct):
    destruct_index = np.random.choice(np.arange(len(wDestruct)), p=np.array(wDestruct) / sum(wDestruct))
    construct_index = np.random.choice(np.arange(len(wConstruct)), p=np.array(wConstruct) / sum(wConstruct))
    destroyed_info = destructOperatorList[destruct_index](current_solution, d_num)
    new_solution = constructOperatorList[construct_index](*destroyed_info, current_solution)
    return new_solution, destruct_index, construct_index



def ALNS(instance_name, iteration_limit=100, duration=3600):
    instance = read_excel(instance_name + ".xlsx")
    timesDestruct = [0 for _ in range(d_operator_num)]       # 使用次数
    timesConstruct = [0 for _ in range(c_operator_num)]
    totalScoreDestruct = [0 for _ in range(d_operator_num)]  # 总的分数
    totalScoreConstruct = [0 for _ in range(c_operator_num)]
    wDestruct = [1 for _ in range(d_operator_num)]           # 破坏算子初始权重
    wConstruct = [1 for _ in range(c_operator_num)]          # 重建算子初始权重

    solution_table = {}

    task_num = len(instance)
    d_num = math.ceil(task_num * d_num_coefficient)  # 破坏个数

    solution = generate_solution_nearest(instance)


    solution_table[solution.hash_key] = solution

    current_fitness = solution.get_fitness()

    best_solution = solution
    best_fitness = current_fitness
    count = 0

    CONSTANT_T = T_coefficient

    start_t = time.time()

    while count <= iteration_limit and time.time() - start_t < duration:
        # if time.time() - start_t > duration:
        #     break

        count += 1
        new_solution, destruct_index, construct_index = destruct_construct(
            solution, d_num, wDestruct, wConstruct
        )

        # is_accept = False

        is_new = False  # 是否是一个之前为探索过的解
        if new_solution.hash_key not in solution_table.keys():
            solution_table[new_solution.hash_key] = new_solution
            is_new = True

        new_fitness = new_solution.get_fitness()

        scoreDestruct = 0
        scoreConstruct = 0

        if new_fitness < current_fitness:
            # is_accept = True
            # p_a = 1.0
            solution = new_solution
            current_fitness = new_fitness
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                scoreDestruct = sigma_1
                scoreConstruct = sigma_1
            else:
                if is_new:
                    scoreDestruct = sigma_2
                    scoreConstruct = sigma_2
        elif new_fitness == current_fitness:
            # is_accept = True
            if is_new:
                scoreDestruct = sigma_2
                scoreConstruct = sigma_2
            # p_a = 0
        else:
            p_a = math.exp((current_fitness - new_fitness) / CONSTANT_T)
            if random.random() < p_a:
                # is_accept = True
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
                if timesDestruct[i] != 0:
                    dTime = timesDestruct[i]
                else:
                    dTime = 1

                wDestruct[i] = wDestruct[i] * (1 - rho) + rho * totalScoreDestruct[i] / dTime
                totalScoreDestruct[i] = 0
                timesDestruct[i] = 0
            for i in range(c_operator_num):
                if timesConstruct[i] != 0:
                    cTime = timesConstruct[i]
                else:
                    cTime = 1
                wConstruct[i] = wConstruct[i] * (1 - rho) + rho * totalScoreConstruct[i] / cTime
                totalScoreConstruct[i] = 0
                timesConstruct[i] = 0


    # fitness_optimal = RR.optimal_solution_dict[instance_name]["fitness"]
    # if best_fitness < fitness_optimal:
    #     RR.optimal_solution_dict[instance_name]["code"] = best_solution.code
    #     RR.optimal_solution_dict[instance_name]["fitness"] = best_solution.fitness
    #     RR.update_optimal_solution()
    elapsed = time.time() - start_t
    termination_reason = "time_limit" if elapsed >= duration else "iteration_limit"
    return best_solution, best_fitness, elapsed, count, termination_reason


# if __name__ == "__main__":
#     RR.init_optimal_solution()
#     instance_name = "N20_K2_M12_I1"
#     solution, best_fitness, run_time = ALNS(instance_name, 100, 10)
#     # preprocess_schedule(solution)
#     print(solution.get_path_map())
#     print(f"run_time:{run_time}")
#     print(f"distance:{solution.distance}  tardiness:{solution.tardiness}")
#     pass
