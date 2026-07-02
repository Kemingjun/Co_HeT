import random

from Util.load_data import read_excel
import math
from Util.generate_init_solution import generate_solution_nearest
import time
from Util.Solution import Solution
from Util.util import *
import copy
import numpy as np
POP_SIZE = 100
employed_size = 10
onLooker_size = 50
LIMIT = 5000
r = 40


class Bee:
    def __init__(self, id, type):
        self.id = id
        self.type = type
    def setType(self, type):
        self.type = type

    def getType(self):
        return self.type

    def getId(self):
        return self.id



class Nectar:
    def __init__(self, solution, fitness=None):
        self.solution = solution
        self.search_num = 0
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
        position = random.sample(feasible_positions, 1)[0]
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

def DABC(instance_name):
    instance = read_excel(instance_name + ".xlsx")
    nectar_list = []
    onlooker_list = []
    scout_list = []
    task_num = len(instance)

    solution = generate_solution_nearest(instance)

    employed_bee = Bee(0, 1)
    nectar = Nectar(solution)
    nectar.setBee(employed_bee)
    nectar_list.append(nectar)


    for i in range(1, employed_size):

        employed_bee = Bee(i, 1)
        # if random.random() < 0.5:
        # else:
        solution = generate_solution_nearest(instance, random.uniform(0.75, 0.95))
        nectar = Nectar(solution)
        nectar.setBee(employed_bee)
        nectar_list.append(nectar)


    for i in range(employed_size, onLooker_size + employed_size):
        onlooker_bee = Bee(i, 2)
        onlooker_list.append(onlooker_bee)

    best_fitness = min([item.fitness for item in nectar_list])
    best_index = [item.fitness for item in nectar_list].index(best_fitness)
    best_nectar = nectar_list[best_index]
    best_solution = best_nectar.solution

    start_t = time.time()
    count = 0

    duration = 1800

    while count <= 100:
    # while time.time() - start_t <= duration:
        count += 1
        new_nectar_list = []
        for nc in nectar_list:
            if time.time() - start_t > duration:
                break
            nc_solution = nc.solution
            nc_fitness = nc.fitness
            new_solution = get_neighbor_solution(nc_solution)
            new_fitness = new_solution.get_fitness()
            if new_fitness < nc_fitness:
                employed_bee = nc.bee
                new_nectar = Nectar(new_solution, new_fitness)
                new_nectar.setBee(employed_bee)
                new_nectar_list.append(new_nectar)
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
                    best_nectar = new_nectar
            else:
                nc.add_search_num()
                if nc.search_num > LIMIT and nc != best_nectar:
                    bee = nc.bee
                    bee.setType(3)
                    scout_list.append(bee)
                else:
                    new_nectar_list.append(nc)
            # else:
            #     new_nectar_list.append(nc)
        nectar_list = copy.deepcopy(new_nectar_list)

        # for _ in range(r):
        #     if len(nectar_list) == 0:
        #         continue
        onlooker_bee_num = len(onlooker_list)
        onlooker_nectar_index = get_index_roulette(nectar_list, onlooker_bee_num)
        new_onlooker_list = []

        for i, onlooker_bee in enumerate(onlooker_list):
            if time.time() - start_t > duration:
                break
            onlooker_nectar = nectar_list[onlooker_nectar_index[i]]
            nectar_solution = onlooker_nectar.solution
            nectar_fitness = onlooker_nectar.fitness
            best_neighbor = None
            min_neighbor_fitness = 1e4
            for _ in range(r):
                new_solution = get_neighbor_solution(nectar_solution)
                new_fitness = new_solution.get_fitness()
                if new_fitness < min_neighbor_fitness:
                    min_neighbor_fitness = new_fitness
                    best_neighbor = new_solution
            if min_neighbor_fitness < nectar_fitness:
                new_nectar = Nectar(best_neighbor, min_neighbor_fitness)
                onlooker_bee.setType(1)
                new_nectar.setBee(onlooker_bee)
                nectar_list[onlooker_nectar_index[i]] = new_nectar
                origin_nectar_bee = onlooker_nectar.bee
                origin_nectar_bee.setType(2)
                new_onlooker_list.append(origin_nectar_bee)
                if min_neighbor_fitness < best_fitness:
                    best_solution = best_neighbor
                    best_fitness = min_neighbor_fitness
                    best_nectar = new_nectar
            else:
                onlooker_nectar.add_search_num(r)
                #     onlooker_nectar

                new_onlooker_list.append(onlooker_bee)

            onlooker_list = copy.deepcopy(new_onlooker_list)
        for scout_bee in scout_list:
            # print("scout_bee")
            new_solution = generate_solution_nearest(instance, random.uniform(0.75, 0.95))
            new_nectar = Nectar(new_solution)
            new_fitness = new_nectar.fitness
            scout_bee.setType(1)
            new_nectar.setBee(scout_bee)
            nectar_list.append(new_nectar)
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                best_nectar = new_nectar
        scout_list = []
    fitness_list = [nectar.fitness for nectar in nectar_list]
    print(sorted(fitness_list))
    for nectar in nectar_list:
        print(f"fitness:{nectar.fitness} search times:{nectar.search_num}")
    return best_fitness




if __name__ == "__main__":
    # init_optimal_solution()
    instance_name = "N100_K2_M12_I1"
    # instance = read_excel(instance_name + ".xlsx")
    # T = get_T(instance)
    DABC(instance_name)
    # update_optimal_solution()
    pass







