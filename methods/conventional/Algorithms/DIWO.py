
from Util.generate_init_solution import generate_solution_nearest
from Util.util import *
from Util.load_data import read_excel
import time
from Util.Solution import Solution
# import Util.result_record as RR


'''入侵杂草算法参数'''
POP_INITIAL_SIZE = 150     # 初始种群大小
POP_MAX_SIZE = 200         # 种群最大大小
S_MAX = 40                # seed最大数
S_MIN = 1                 # seed最小数


def neighbor_insertion(solution):
    task = random.randint(1, solution.task_num)
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()

    remove_(sequence_map, path_init_task_map, task)

    for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
        feasible_positions = get_feasible_insert_position(sequence_map, path_init_task_map, task, robot_type)
        position = random.choice(list(feasible_positions))
        insert_(sequence_map, path_init_task_map, task, position, robot_type)
    return Solution(solution.instance, sequence_map, path_init_task_map)

def get_task_position(path_map, task):
    position = []
    for path_index, path in path_map.items():
        for ind, _task in enumerate(path):
            if _task == task:
                position.append((path_index, ind))
    return position

# def remove_task_from_path_map(path_map, task):
#     for path_index, path in path_map.items():
#         if task in path:
#             path.remove(task)

def neighbor_swap(solution):
    task_list = list(range(1, solution.task_num + 1))
    task_1 = random.sample(task_list, 1)[0]
    task_list.remove(task_1)
    task_2 = random.sample(task_list, 1)[0]

    path_map = solution.get_path_map()

    task_1_positions = get_task_position(path_map, task_1)
    task_2_positions = get_task_position(path_map, task_2)


    # remove_task_from_path_map(path_map, task_1)
    # remove_task_from_path_map(path_map, task_2)

    for position in task_1_positions:
        path_map[position[0]][position[1]] = task_2

    for position in task_2_positions:
        path_map[position[0]][position[1]] = task_1

    sequence_map, path_init_task_map = path_map2sequence_map(path_map)

    return Solution(solution.instance, sequence_map, path_init_task_map)

def get_neighbor_solution(solution):
    neighbor_list = [neighbor_insertion, neighbor_swap]
    neighbor_index = random.randint(0,1)
    neighbor_solution = neighbor_list[neighbor_index](solution)
    return neighbor_solution


def DIWO(instance_name, iteration_limit=100, duration=3600):
    instance = read_excel(instance_name + ".xlsx")
    # task_num = len(instance)

    weed_list = []

    best_solution = generate_solution_nearest(instance)
    best_fitness = best_solution.get_fitness()

    weed_list.append(best_solution)

    for _ in range(POP_INITIAL_SIZE - 1):
        init_solution = generate_solution_nearest(instance, random.uniform(0.75, 0.95))
        weed_list.append(init_solution)

    start_t = time.time()
    count = 0

    # while time.time() - start_time < duration:
    while count <= iteration_limit and time.time() - start_t < duration:
        count += 1
        fitness_list = [solution.get_fitness() for solution in weed_list]
        min_fitness = min(fitness_list)
        max_fitness = max(fitness_list)

        # 确保 best_solution 始终是当前找到的最佳解决方案
        # 如果当前 min_fitness 比之前记录的 best_fitness 更好，就更新 best_solution
        if min_fitness < best_fitness: # 或者使用 <= 取决于你如何定义“更好”
            best_fitness = min_fitness
            # 这里需要找到对应 min_fitness 的 solution 对象
            # 由于 weed_list 是经过排序的，best_solution 始终是 weed_seed_list[0]
            # 但是为了避免在循环中间遗漏，可以再确认一下
            # 实际上，在每次迭代结束时，best_solution = weed_seed_list[0] 已经更新了
            # 所以我们只需要确保在函数结束时返回的 best_solution 是最终的那个
            pass # 这一行可以不用改变，因为后面的 weed_seed_list[:POP_MAX_SIZE] 会保留最好的

        seed_list = []

        for weed_solution in weed_list:
            weed_fitness = weed_solution.get_fitness()

            '''获得每个weed产生seed的个数'''
            try:
                if abs(max_fitness - min_fitness) < 1e-5:
                    seed_num = random.randint(S_MIN, S_MAX)
                else:
                    seed_num = math.floor(S_MAX - (weed_fitness - min_fitness) / (max_fitness - min_fitness) * \
                                          (S_MAX - S_MIN))
            except:
                seed_num = random.randint(S_MIN, S_MAX)

            for seed_index in range(seed_num):
                new_solution = get_neighbor_solution(weed_solution)
                seed_list.append(new_solution)

        weed_seed_list = seed_list + weed_list
        weed_seed_list = sorted(weed_seed_list, key=lambda x: x.get_fitness())

        # 每次迭代结束后，更新 best_solution 为当前找到的最佳 solution 对象
        best_solution = weed_seed_list[0]

        weed_list = weed_seed_list[:POP_MAX_SIZE]

        # print(f"第：{count} time: {time.time() - start_t} best_fitness: {best_fitness}")
    # print(sorted(fitness_list))

    # Return the best solution.
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
#     # instance = read_excel(instance_name + ".xlsx")
#     # T = get_T(instance)
#     best_solution, best_fitness, run_time = DIWO(instance_name)
#     print(f"best_fitness:{best_fitness} run_time:{run_time}")
#     # update_optimal_solution()
#     pass
