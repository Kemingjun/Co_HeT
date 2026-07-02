import sys
import os
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.dirname(current_dir)

sys.path.append(root_dir)
# sys.path.append('C:/Users/13360/Desktop/HRSP/HRSP_ALNS/HRPS_ALNS')

from gurobipy import *
from Util.load_data import read_excel
from Util.util import *
from Util.Solution import Solution


def build_model(instance_name, results=None):
    if results == None:
        results = {}

    result = {}
    LARGE = 100
    logging.info(f"---------------------------- {instance_name} start ----------------------------")
    logging.info(f"instance: {instance_name}")
    print(f"---------------------------- {instance_name} start ----------------------------")
    print(f"instance: {instance_name}")
    instance = read_excel(instance_name)

    task_num = len(instance)

    x_ijr_index = {}  # edge
    A_ik_index = {}
    C_ik_index = {}
    S_i_index = {}
    F_i_index = {}
    T_i_index = {}


    distance_map = {}
    for i in range(task_num + 1):
        for j in range(task_num + 1):
            if i == 0:
                i_position = [0, 0]
            else:
                i_position = [instance[i - 1][1], instance[i - 1][2]]
            if j == 0:
                j_position = [0, 0]
            else:
                j_position = [instance[j - 1][1], instance[j - 1][2]]

            distance_map[i, j] = get_distance(i_position, j_position)

    for i in range(0, task_num + 1):
        for k in range(1, Config.ROBOT_TYPE_NUM + 1):
            A_ik_index[i, k] = 0
            C_ik_index[i, k] = 0
        S_i_index[i] = 0
        F_i_index[i] = 0
        T_i_index[i] = 0

    for i in range(task_num + 1):
        for j in range(task_num + 1):
            for r in range(1, Config.ROBOT_NUM + 1):
                x_ijr_index[i, j, r] = 0

    model = Model()
    x = model.addVars(x_ijr_index.keys(), vtype=GRB.BINARY, name='x')
    A = model.addVars(A_ik_index.keys(), vtype=GRB.CONTINUOUS, name='A')
    C = model.addVars(C_ik_index.keys(), vtype=GRB.CONTINUOUS, name='C')
    S = model.addVars(S_i_index.keys(), vtype=GRB.CONTINUOUS, name='S')
    F = model.addVars(F_i_index.keys(), vtype=GRB.CONTINUOUS, name='F')
    T = model.addVars(T_i_index.keys(), vtype=GRB.CONTINUOUS, name='T')

    distance = model.addVar(vtype=GRB.CONTINUOUS, name='distance')
    tardiness = model.addVar(vtype=GRB.CONTINUOUS, name='tardiness')

    model.addConstr(distance == quicksum(
        x[i, j, r] * distance_map[i, j] for i in range(task_num + 1) for j in range(1, task_num + 1) for r in range(1, Config.ROBOT_NUM + 1)
    ))
    model.addConstr(tardiness == quicksum(T[i] for i in range(1, task_num + 1)))

    model.setObjective(distance * Config.WEIGHT + tardiness * (1 - Config.WEIGHT), GRB.MINIMIZE)
    for i in range(1, task_num + 1):
        for j in range(1, task_num + 1):
            for r in range(1, Config.ROBOT_NUM + 1):
                if i == j:
                    model.addConstr(x[i, j, r] == 0, f"c0_{i}_{j}_{r}")
    for type_index in Config.TYPE_LIST:
        # for r in range(type_index[0], type_index[1]):
        for j in range(1, task_num + 1):
            model.addConstr(quicksum(x[i, j, r] for i in range(0, task_num + 1) for r in range(type_index[0], type_index[1])) == 1, f"c1_{j}_{type_index}")

    for type_index in Config.TYPE_LIST:
        for i in range(1, task_num + 1):
            model.addConstr(quicksum(x[i, j, r] for j in range(0, task_num + 1) for r in range(type_index[0], type_index[1])) == 1, f"c2_{i}_{type_index}")
    for j in range(1, task_num + 1):
        for r in range(1, Config.ROBOT_NUM + 1):
            model.addConstr(
                (quicksum(x[i, j, r] for i in range(0, task_num + 1)) - quicksum(x[j, i, r] for i in range(0, task_num + 1))) == 0,
                f'c3_{j}_{r}'
            )

    # for r in range(1, Config.ROBOT_NUM + 1):
    #     model.addConstr(quicksum(x[0, i, r] for i in range(1, task_num + 1)) == quicksum(x[i, 0, r] for i in range(1, task_num + 1)),
    #                     f'c4_{r}')
    for r in range(1, Config.ROBOT_NUM + 1):
        model.addConstr(quicksum(x[0, i, r] for i in range(0, task_num + 1)) == 1,
                        f'c5_{r}')

    for r in range(1, Config.ROBOT_NUM + 1):
        model.addConstr(quicksum(x[i, 0, r] for i in range(0, task_num + 1)) == 1,
                        f'c6_{r}')
    for k in range(1, Config.ROBOT_TYPE_NUM + 1):
        model.addConstr(C[0, k] == 0, f'c7_{k}')
    for _k, type_index in enumerate(Config.TYPE_LIST):
        k = _k + 1
        for r in range(type_index[0], type_index[1]):
            for j in range(1, task_num + 1):
                for i in range(0, task_num + 1):
                    model.addConstr(C[i, k] + distance_map[i, j] - LARGE * (1 - x[i, j, r]) <= A[j, k], f"c8_{k}_{r}_{j}_{i}")

    # for _k, type_index in enumerate(Config.TYPE_LIST):
    #     k = _k + 1
    #     for i in range(1, task_num + 1):
    #         model.addConstr(A[i, k] <= S[i], f'c9_{k}_{i}')
    for _k, type_index in enumerate(Config.TYPE_LIST):
        k = _k + 1
        for i in range(1, task_num + 1):
            model.addConstr(A[i, k] <= S[i], f'c10_{k}_{i}')
            model.addConstr((S[i] + instance[i - 1][4][_k]) <= C[i, k], f'c11_{k}_{i}')
            model.addConstr(C[i, k] <= F[i], f'c12_{k}_{i}')
    for i in range(1, task_num + 1):
        model.addConstr(F[i] - instance[i - 1][3] <= T[i], f'c13_{i}')
        model.addConstr(0 <= T[i], f'c14_{i}')

    model.setParam("TimeLimit", 3600)
    model.setParam("LogFile", "gurobi_log.txt")

    model.optimize()
    if model.SolCount == 0:
        result['instance'] = instance_name
        result['fitness'] = " "
        result['distance'] = " "
        result['tardiness'] = " "
        result['solution'] = " "

        results[instance_name] = result
        return None

    solution = get_path_list(x, instance)
    print(f"distance: {distance.x} tardiness: {tardiness.x}")

    for r in range(1, Config.ROBOT_NUM + 1):
        route_distance = 0
        for i in range(0, task_num + 1):
            for j in range(1, task_num + 1):
                if round(x[i, j, r].x, 0) == 1:
                    route_distance += distance_map[i, j]
        print(f"route{r}: {route_distance}")



    result['instance'] = instance_name
    result['fitness'] = solution.get_fitness()
    result['distance'] = solution.distance
    result['tardiness'] = solution.tardiness
    result['solution'] = str(solution.get_path_map())
    result['solve_time'] = model.Runtime

    results[instance_name] = result


    return solution


def get_path_list(x, instance):


    path_map = {}
    task_num = len(instance)




    for r in range(1, Config.ROBOT_NUM + 1):
        flag = False
        path = []
        connection_dict = {}
        for i in range(0, task_num + 1):
            for j in range(0, task_num + 1):
                if round(x[i, j, r].x, 0) == 1:
                    connection_dict[i] = j
                    flag = True
        if not flag:
            # logging.info(f"route{r} connection_dict:{connection_dict}")
            logging.info(f"route{r} path:{path}")
            path_map[r] = []
            continue
        current = 0
        path.append(0)
        while True:
            current = connection_dict.get(current)
            if current == 0:
                break
            path.append(current)
        path.remove(0)
        logging.info(f"route{r} path:{path}")

        path_map[r] = path
    print(path_map)
    sequence_map, path_init_task_map = path_map2sequence_map(path_map)
    solution = Solution(instance, sequence_map, path_init_task_map)
    return solution









if __name__ == "__main__":
    results = {}
    size = 10
    for i in range(5, 21):
        instance_name = f"N{size}_K2_M12_I{i}.xlsx"
        solution = build_model(instance_name, results)
        df = pd.DataFrame(results).T
        df.to_excel(f'results_{size}.xlsx', index=False)
        fitness = solution.get_fitness()
        print(f"solution fitness:{fitness} distance:{solution.distance} tardiness:{solution.tardiness}")


    # instance_name = f"N50_K2_M12_I1.xlsx"
    # solution = build_model(instance_name)













