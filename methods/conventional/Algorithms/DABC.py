import random

from Util.load_data import read_excel
import math
from Util.generate_init_solution import generate_solution_nearest
import time
from Util.Solution import Solution
from Util.util import *
import copy
import numpy as np
# import Util.result_record as RR


'''人工蜂群算法参数'''
# max_iteration = 800  # 最大迭代次数
POP_SIZE = 100  # 蜂群的总数
employed_size = 10  # 雇佣蜂个数
onLooker_size = 50  # 跟随蜂个数
LIMIT = 5000  # 蜜源被限制的搜索次数
r = 40  # 跟随蜂重复次数


class Bee:
    def __init__(self, id, type):
        self.id = id
        self.type = type
        '''
        type 1: 雇佣蜂 employed Bee
        type 2: 追随蜂 Onlooker Bee
        type 3: 侦查蜂 Scout Bee
        '''

    def setType(self, type):
        self.type = type

    def getType(self):
        return self.type

    def getId(self):
        return self.id



class Nectar:
    def __init__(self, solution, fitness=None):
        self.solution = solution  # 当前蜜源对应的解
        self.search_num = 0  # 被搜索的次数
        # self.followers = []         # 跟随蜂
        self.fitness = fitness if fitness is not None else solution.get_fitness()
        self.bee = None

    def setBee(self, bee):
        self.bee = bee

    def add_search_num(self, time=None):
        if time is not None:
            self.search_num += time
        else:
            self.search_num += 1

    def getBee(self):
        return self.bee


def get_neighbor_solution(solution):
    neighbor_list = [neighbor_insertion, neighbor_swap]
    neighbor_index = random.randint(0,len(neighbor_list) - 1)
    neighbor_solution = neighbor_list[neighbor_index](solution)
    return neighbor_solution

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

def get_index_roulette(nectar_list, num):
    cost = np.array([nc.fitness for nc in nectar_list])
    for j in range(len(cost)):
        cost[j] = - cost[j]
    fitness_list = (cost - np.min(cost)) + 1e-3
    idx = np.random.choice(np.arange(len(nectar_list)), size=num, replace=True,
                           p=(fitness_list) / (fitness_list.sum()))
    return idx

def DABC(instance_name, iteration_limit=100, duration=3600):
    instance = read_excel(instance_name + ".xlsx")
    nectar_list = []  # 蜜源集合，即雇佣蜂集合
    onlooker_list = []  # 追随蜂集合
    scout_list = []  # 侦查蜂集合
    # task_num = len(instance)

    solution = generate_solution_nearest(instance)

    employed_bee = Bee(0, 1)
    nectar = Nectar(solution)  # 得到一个蜜源
    nectar.setBee(employed_bee)  # 每一个蜜源对应一个雇佣蜂
    nectar_list.append(nectar)


    for i in range(1, employed_size):

        employed_bee = Bee(i, 1)
        # if random.random() < 0.5:
        #     solution = generate_solution_random(instance)  # 获得一个可行解
        # else:
        solution = generate_solution_nearest(instance, random.uniform(0.75, 0.95))
        nectar = Nectar(solution)  # 得到一个蜜源
        nectar.setBee(employed_bee)  # 每一个蜜源对应一个雇佣蜂
        nectar_list.append(nectar)
        # print(f"生成第{i}个employed bee fitness:{solution.get_fitness()}")


    for i in range(employed_size, onLooker_size + employed_size):
        onlooker_bee = Bee(i, 2)
        onlooker_list.append(onlooker_bee)

    best_fitness = min([item.fitness for item in nectar_list])
    best_index = [item.fitness for item in nectar_list].index(best_fitness)
    best_nectar = nectar_list[best_index]
    best_solution = best_nectar.solution

    start_t = time.time()
    count = 0

    # duration = 1800

    while count <= iteration_limit and time.time() - start_t < duration:
    # while time.time() - start_t <= duration:
        count += 1
        '''雇佣蜂阶段'''
        new_nectar_list = []
        for nc in nectar_list:
            if time.time() - start_t > duration:
                break
            nc_solution = nc.solution
            nc_fitness = nc.fitness
            new_solution = get_neighbor_solution(nc_solution)
            new_fitness = new_solution.get_fitness()
            if new_fitness < nc_fitness:
                # 原来的蜜源就被舍弃了
                employed_bee = nc.bee
                new_nectar = Nectar(new_solution, new_fitness)
                new_nectar.setBee(employed_bee)
                new_nectar_list.append(new_nectar)
                # 更新最优解
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
                    best_nectar = new_nectar
            else:
                nc.add_search_num()
                '''不动best_nectar'''
                if nc.search_num > LIMIT and nc != best_nectar:
                    # 这个蜜源的探索次数太多了
                    bee = nc.bee
                    bee.setType(3)  # 该蜜蜂成为侦查蜂
                    scout_list.append(bee)
                    # print("探索次数过多，变为侦查蜂")
                else:
                    new_nectar_list.append(nc)
            # else:
            #     new_nectar_list.append(nc)
        nectar_list = copy.deepcopy(new_nectar_list)  # 蜂源更新

        # for _ in range(r):
        #     if len(nectar_list) == 0:
        #         print(f"nectar_list为空")
        #         continue

        '''跟随蜂阶段'''
        onlooker_bee_num = len(onlooker_list)
        onlooker_nectar_index = get_index_roulette(nectar_list, onlooker_bee_num)  # 根据轮盘赌选择出跟随蜂对应的蜜源
        new_onlooker_list = []

        for i, onlooker_bee in enumerate(onlooker_list):
            if time.time() - start_t > duration:
                break
            '''选择一个雇佣蜂'''
            onlooker_nectar = nectar_list[onlooker_nectar_index[i]]  # 当前跟随蜂选择的蜜源
            nectar_solution = onlooker_nectar.solution
            nectar_fitness = onlooker_nectar.fitness
            '''产生新解'''
            best_neighbor = None
            min_neighbor_fitness = 1e4
            for _ in range(r):
                new_solution = get_neighbor_solution(nectar_solution)
                new_fitness = new_solution.get_fitness()
                if new_fitness < min_neighbor_fitness:
                    min_neighbor_fitness = new_fitness
                    best_neighbor = new_solution
            if min_neighbor_fitness < nectar_fitness:
                # 跟随蜂找到的解更好，原雇佣蜂变为跟随蜂，跟随蜂变为雇佣蜂
                new_nectar = Nectar(best_neighbor, min_neighbor_fitness)
                onlooker_bee.setType(1)  # 变为雇佣蜂
                new_nectar.setBee(onlooker_bee)
                nectar_list[onlooker_nectar_index[i]] = new_nectar
                '''原雇佣蜂变为跟随蜂'''
                origin_nectar_bee = onlooker_nectar.bee
                origin_nectar_bee.setType(2)  # 原跟随蜂变为雇佣蜂
                new_onlooker_list.append(origin_nectar_bee)
                '''更新最优解'''
                if min_neighbor_fitness < best_fitness:
                    best_solution = best_neighbor
                    best_fitness = min_neighbor_fitness
                    best_nectar = new_nectar
            else:
                '''原蜜源被探索次数加1'''
                onlooker_nectar.add_search_num(r)
                # if onlooker_nectar.search_num > LIMIT:   # 探索次数超过限制
                #     '''该蜜源对应的雇佣蜂成为侦查蜂'''
                #     onlooker_nectar

                new_onlooker_list.append(onlooker_bee)

            onlooker_list = copy.deepcopy(new_onlooker_list)

        '''侦查蜂阶段'''
        for scout_bee in scout_list:
            # print(f"侦查蜂阶段")
            # print("scout_bee")
            new_solution = generate_solution_nearest(instance, random.uniform(0.75, 0.95))  # ILS产生一个新的蜜源
            new_nectar = Nectar(new_solution)
            new_fitness = new_nectar.fitness
            scout_bee.setType(1)  # 原侦查蜂变为雇佣蜂
            new_nectar.setBee(scout_bee)
            nectar_list.append(new_nectar)
            '''更新最优解'''
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                best_nectar = new_nectar
        scout_list = []  # 清空侦查蜂
        # print(f"第{count}代，最优解为：{round(best_fitness, 8)} time:{time.time() - start_t}")
    # fitness_list = [nectar.fitness for nectar in nectar_list]
    # print(sorted(fitness_list))
    # for nectar in nectar_list:
    #     print(f"fitness:{nectar.fitness} search times:{nectar.search_num}")

    # fitness_optimal = RR.optimal_solution_dict[instance_name]["fitness"]
    # if best_fitness < fitness_optimal:
    #     RR.optimal_solution_dict[instance_name]["code"] = best_solution.code
    #     RR.optimal_solution_dict[instance_name]["fitness"] = best_solution.fitness
    #     RR.update_optimal_solution()

    elapsed = time.time() - start_t
    termination_reason = "time_limit" if elapsed >= duration else "iteration_limit"
    return best_solution, best_fitness, elapsed, count, termination_reason



# if __name__ == "__main__":
#     # init_optimal_solution()
#     instance_name = "N20_K2_M12_I1"
#     # instance = read_excel(instance_name + ".xlsx")
#     # T = get_T(instance)
#     best_solution, best_fitness, run_time = DABC(instance_name)
#     print(f"best_fitness:{best_fitness} run_time:{run_time}")
#     # update_optimal_solution()
#     pass





