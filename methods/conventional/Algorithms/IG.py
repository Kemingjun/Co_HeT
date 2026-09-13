
from Util.load_data import read_excel
import math
from Util.generate_init_solution import generate_solution_random, generate_solution_nearest
import time
from Util.operators import destroy_random, repair_greedy
import random
from Util.Solution import Solution
from Util.util import *
# import Util.result_record as RR

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
        for task in task_list:  # 每个任务都greedy搜索一遍
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
    优化不重新开始
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
    for task in task_list:  # 每个任务都greedy搜索一遍
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
    return current_solution






def IGA(instance_name, iteration_limit=100, duration=3600):
    instance = read_excel(instance_name + ".xlsx")
    task_num = len(instance)
    d_num = math.ceil(task_num * d_num_coefficient)
    # init_start = time.time()
    solution = generate_solution_nearest(instance)
    # init_end = time.time()
    # print(f"初始解生成时间: {init_end - init_start}")

    current_fitness = solution.get_fitness()

    best_solution = solution
    best_fitness = current_fitness

    start_t = time.time()

    count = 0

    while count <= iteration_limit and time.time() - start_t < duration:
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

        # print(f"第{count}次迭代，最优fitness为：{round(best_fitness, 4)} 当前fitness为 ：{round(current_fitness, 4)} 新fitness：{round(new_fitness, 4)}  time:{round(time.time() - start_t, 4)}")
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
#     solution, best_fitness, run_time = IGA(instance_name, 100, 10)
#     # preprocess_schedule(solution)
#     print(solution.get_path_map())
#     print(f"run_time:{run_time}")
#     print(f"distance:{solution.distance}  tardiness:{solution.tardiness}")
#     print(f"fitness:{best_fitness} {solution.get_fitness()}")
#     pass



