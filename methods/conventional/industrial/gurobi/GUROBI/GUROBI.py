import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from gurobipy import GRB, Model, quicksum

from Util.load_data import read_excel
from Util.Solution import Solution
from Util.util import (
    Config,
    deadline,
    delivery_pos,
    get_distance,
    handover_pos,
    path_map2sequence_map,
    supply_pos,
)


STATUS_NAMES = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.CUTOFF: "CUTOFF",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.NUMERIC: "NUMERIC",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.INPROGRESS: "INPROGRESS",
    GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
}


def _type_of_robot(robot):
    for robot_type, (low, high) in enumerate(Config.TYPE_LIST, start=1):
        if low <= robot < high:
            return robot_type
    raise ValueError(f"Unknown robot id: {robot}")


def _distance(instance, from_task, to_task, robot_type):
    if robot_type == Config.FORKLIFT_TYPE:
        from_pos = Config.DEPOT if from_task == 0 else handover_pos(instance[from_task - 1])
        to_pos = supply_pos(instance[to_task - 1])
        return get_distance(from_pos, to_pos)
    from_pos = Config.DEPOT if from_task == 0 else delivery_pos(instance[from_task - 1])
    to_pos = delivery_pos(instance[to_task - 1])
    return get_distance(from_pos, to_pos)


def _delivery_or_depot(instance, task):
    if task == 0:
        return Config.DEPOT
    return delivery_pos(instance[task - 1])


def _handover_or_depot(instance, task):
    if task == 0:
        return Config.DEPOT
    return handover_pos(instance[task - 1])


def _build_path_map(x, instance):
    task_num = len(instance)
    path_map = {}
    for robot in range(1, Config.ROBOT_NUM + 1):
        connection = {}
        for i in range(task_num + 1):
            for j in range(task_num + 1):
                if i != j and round(x[i, j, robot].x) == 1:
                    connection[i] = j
        path = []
        current = 0
        while connection.get(current, 0) != 0:
            current = connection[current]
            path.append(current)
        path_map[robot] = path
    return path_map


def _status_name(status):
    return STATUS_NAMES.get(status, f"UNKNOWN_{status}")


def _format_solution_report(result):
    objective_gap = None
    if isinstance(result.get("solver_objective"), (int, float)) and isinstance(result.get("fitness"), (int, float)):
        objective_gap = abs(result["solver_objective"] - result["fitness"])
    lines = [
        "========== Gurobi Solution Report ==========",
        f"Instance: {result.get('instance')}",
        f"Status: {result.get('status_name')} ({result.get('status')})",
        f"Runtime: {result.get('runtime')}",
        f"SolCount: {result.get('sol_count')}",
        f"Best bound: {result.get('best_bound')}",
        f"MIP gap: {result.get('mip_gap')}",
        f"Solver objective: {result.get('solver_objective')}",
        f"Unified fitness: {result.get('fitness')}",
        f"Objective consistency gap: {objective_gap}",
        f"Unified distance: {result.get('distance')}",
        f"Unified travel time: {result.get('travel_time')}",
        f"Unified tardiness: {result.get('tardiness')}",
        "Path map:",
        str(result.get("solution")),
        "==========================================",
    ]
    return "\n".join(lines)


def _print_model_summary(instance_name, task_num, time_limit):
    print("========== Gurobi Build Summary ==========")
    print(f"Instance: {instance_name}")
    print(f"Tasks: {task_num}")
    print(f"Robot types: {Config.ROBOT_TYPE_NUM}")
    print(f"Robots: {Config.ROBOT_NUM} {Config.ROBOT_NUM_LIST}")
    print(f"Time limit: {time_limit}")
    print(f"Objective: {Config.WEIGHT} * travel_time + {1 - Config.WEIGHT} * tardiness")
    print("=========================================")


def build_model(instance_name, results=None, time_limit=60, verbose=True, gurobi_threads=8):
    if results is None:
        results = {}

    logging.info("start Gurobi industrial simulation instance %s", instance_name)
    instance = read_excel(instance_name)
    task_num = len(instance)
    if verbose:
        _print_model_summary(instance_name, task_num, time_limit)

    tasks = range(1, task_num + 1)
    nodes = range(0, task_num + 1)
    robots = range(1, Config.ROBOT_NUM + 1)

    model = Model("real_world_chrsp")
    x_index = [
        (i, j, r)
        for i in nodes
        for j in nodes
        for r in robots
        if i != j
    ]
    x = model.addVars(x_index, vtype=GRB.BINARY, name="x")
    u = model.addVars(tasks, robots, lb=0, ub=task_num, vtype=GRB.CONTINUOUS, name="u")

    tc = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="carrier_release")
    ts = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="shuttle_release")
    tf = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="forklift_release")
    carrier_arrive_ad = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="carrier_arrive_ad")
    attach_start = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="attach_start")
    attach_end = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="attach_end")
    carrier_shuttle_arrive_handover = model.addVars(
        tasks, lb=0, vtype=GRB.CONTINUOUS, name="carrier_shuttle_arrive_handover"
    )
    forklift_arrive_supply = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="forklift_arrive_supply")
    forklift_depart_supply = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="forklift_depart_supply")
    forklift_arrive_handover = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="forklift_arrive_handover")
    handover_start = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="handover_start")
    handover_end = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="handover_end")
    arrive_destination = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="arrive_destination")
    tardiness = model.addVars(tasks, lb=0, vtype=GRB.CONTINUOUS, name="tardiness")
    total_distance = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="distance")
    total_travel_time = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="travel_time")

    for robot_type, (low, high) in enumerate(Config.TYPE_LIST, start=1):
        type_robots = range(low, high)
        for j in tasks:
            model.addConstr(
                quicksum(x[i, j, r] for i in nodes if i != j for r in type_robots) == 1,
                f"assign_in_{robot_type}_{j}",
            )
            model.addConstr(
                quicksum(x[j, k, r] for k in nodes if k != j for r in type_robots) == 1,
                f"assign_out_{robot_type}_{j}",
            )

    for r in robots:
        model.addConstr(quicksum(x[0, j, r] for j in tasks) <= 1, f"start_{r}")
        model.addConstr(quicksum(x[i, 0, r] for i in tasks) <= 1, f"end_{r}")
        for h in tasks:
            model.addConstr(
                quicksum(x[i, h, r] for i in nodes if i != h)
                == quicksum(x[h, j, r] for j in nodes if j != h),
                f"flow_{h}_{r}",
            )
            model.addConstr(u[h, r] <= task_num * quicksum(x[i, h, r] for i in nodes if i != h))

    for i in tasks:
        for j in tasks:
            if i == j:
                continue
            for r in robots:
                model.addConstr(
                    u[i, r] - u[j, r] + task_num * x[i, j, r] <= task_num - 1,
                    f"mtz_{i}_{j}_{r}",
                )

    large_m = 50000.0
    carrier_robots = range(Config.TYPE_LIST[Config.CARRIER_TYPE - 1][0], Config.TYPE_LIST[Config.CARRIER_TYPE - 1][1])
    shuttle_robots = range(Config.TYPE_LIST[Config.SHUTTLE_TYPE - 1][0], Config.TYPE_LIST[Config.SHUTTLE_TYPE - 1][1])
    forklift_robots = range(Config.TYPE_LIST[Config.FORKLIFT_TYPE - 1][0], Config.TYPE_LIST[Config.FORKLIFT_TYPE - 1][1])

    carrier_in = {
        (i, j): quicksum(x[i, j, r] for r in carrier_robots)
        for j in tasks
        for i in nodes
        if i != j
    }
    shuttle_in = {
        (i, j): quicksum(x[i, j, r] for r in shuttle_robots)
        for j in tasks
        for i in nodes
        if i != j
    }
    forklift_in = {
        (i, j): quicksum(x[i, j, r] for r in forklift_robots)
        for j in tasks
        for i in nodes
        if i != j
    }

    predecessor_pair_index = [
        (carrier_pre, shuttle_pre, task)
        for task in tasks
        for carrier_pre in nodes
        for shuttle_pre in nodes
        if carrier_pre != task and shuttle_pre != task
    ]
    predecessor_pair = model.addVars(predecessor_pair_index, vtype=GRB.BINARY, name="carrier_shuttle_predecessor_pair")

    distance_terms = []
    travel_time_terms = []
    for task in tasks:
        task_info = instance[task - 1]
        supply_position = supply_pos(task_info)
        handover_position = handover_pos(task_info)
        delivery_position = delivery_pos(task_info)
        handover_to_delivery_distance = get_distance(handover_position, delivery_position)
        supply_to_handover_distance = get_distance(supply_position, handover_position)

        model.addConstr(
            quicksum(predecessor_pair[carrier_pre, shuttle_pre, task]
                     for carrier_pre in nodes if carrier_pre != task
                     for shuttle_pre in nodes if shuttle_pre != task) == 1,
            f"single_carrier_shuttle_predecessor_pair_{task}",
        )

        for carrier_pre in nodes:
            if carrier_pre == task:
                continue
            for shuttle_pre in nodes:
                if shuttle_pre == task:
                    continue
                pair = predecessor_pair[carrier_pre, shuttle_pre, task]
                model.addConstr(pair <= carrier_in[carrier_pre, task])
                model.addConstr(pair <= shuttle_in[shuttle_pre, task])
                model.addConstr(pair >= carrier_in[carrier_pre, task] + shuttle_in[shuttle_pre, task] - 1)

                carrier_pre_release = 0 if carrier_pre == 0 else tc[carrier_pre]
                carrier_pre_position = _delivery_or_depot(instance, carrier_pre)
                shuttle_ad_position = _delivery_or_depot(instance, shuttle_pre)
                carrier_to_shuttle_distance = get_distance(carrier_pre_position, shuttle_ad_position)
                model.addConstr(
                    carrier_arrive_ad[task]
                    >= carrier_pre_release
                    + carrier_to_shuttle_distance / Config.CARRIER_VELOCITY
                    - large_m * (1 - pair),
                    f"carrier_arrive_ad_{carrier_pre}_{shuttle_pre}_{task}",
                )
                distance_terms.append(pair * carrier_to_shuttle_distance)
                travel_time_terms.append(pair * carrier_to_shuttle_distance / Config.CARRIER_VELOCITY)

        for shuttle_pre in nodes:
            if shuttle_pre == task:
                continue
            shuttle_pre_release = 0 if shuttle_pre == 0 else ts[shuttle_pre]
            shuttle_ad_position = _delivery_or_depot(instance, shuttle_pre)
            shuttle_to_handover_distance = get_distance(shuttle_ad_position, handover_position)
            model.addConstr(
                attach_start[task]
                >= shuttle_pre_release - large_m * (1 - shuttle_in[shuttle_pre, task]),
                f"attach_wait_shuttle_{shuttle_pre}_{task}",
            )
            model.addConstr(
                carrier_shuttle_arrive_handover[task]
                >= attach_end[task]
                + shuttle_to_handover_distance / Config.CARRIER_VELOCITY
                - large_m * (1 - shuttle_in[shuttle_pre, task]),
                f"carrier_shuttle_arrive_handover_{shuttle_pre}_{task}",
            )
            distance_terms.append(shuttle_in[shuttle_pre, task] * shuttle_to_handover_distance)
            travel_time_terms.append(
                shuttle_in[shuttle_pre, task] * shuttle_to_handover_distance / Config.CARRIER_VELOCITY
            )

        for forklift_pre in nodes:
            if forklift_pre == task:
                continue
            forklift_pre_release = 0 if forklift_pre == 0 else tf[forklift_pre]
            forklift_pre_position = _handover_or_depot(instance, forklift_pre)
            forklift_to_supply_distance = get_distance(forklift_pre_position, supply_position)
            model.addConstr(
                forklift_arrive_supply[task]
                >= forklift_pre_release
                + forklift_to_supply_distance / Config.FORKLIFT_VELOCITY
                - large_m * (1 - forklift_in[forklift_pre, task]),
                f"forklift_arrive_supply_{forklift_pre}_{task}",
            )
            distance_terms.append(forklift_in[forklift_pre, task] * forklift_to_supply_distance)
            travel_time_terms.append(
                forklift_in[forklift_pre, task] * forklift_to_supply_distance / Config.FORKLIFT_VELOCITY
            )

        distance_terms.append(handover_to_delivery_distance)
        distance_terms.append(supply_to_handover_distance)
        travel_time_terms.append(handover_to_delivery_distance / Config.CARRIER_VELOCITY)
        travel_time_terms.append(supply_to_handover_distance / Config.FORKLIFT_VELOCITY)

        model.addConstr(attach_start[task] >= carrier_arrive_ad[task], f"attach_wait_carrier_{task}")
        model.addConstr(attach_end[task] == attach_start[task] + Config.CARRIER_SHUTTLE_COUPLING_TIME)
        model.addConstr(forklift_depart_supply[task] == forklift_arrive_supply[task] + Config.FORKLIFT_PICKUP_TIME)
        model.addConstr(
            forklift_arrive_handover[task]
            == forklift_depart_supply[task] + supply_to_handover_distance / Config.FORKLIFT_VELOCITY
        )
        model.addConstr(handover_start[task] >= carrier_shuttle_arrive_handover[task])
        model.addConstr(handover_start[task] >= forklift_arrive_handover[task])
        model.addConstr(handover_end[task] == handover_start[task] + Config.SOURCE_HANDOVER_TIME)
        model.addConstr(tf[task] == handover_end[task], f"forklift_release_{task}")
        model.addConstr(
            arrive_destination[task]
            == handover_end[task] + handover_to_delivery_distance / Config.CARRIER_VELOCITY
        )
        model.addConstr(
            tc[task] == arrive_destination[task] + Config.CARRIER_SHUTTLE_DECOUPLING_TIME,
            f"carrier_release_{task}",
        )
        model.addConstr(
            ts[task] == tc[task] + Config.SHUTTLE_UNLOADING_TIME + Config.DELIVERY_STATION_PROCESSING_TIME,
            f"shuttle_release_{task}",
        )
        model.addConstr(tardiness[task] >= ts[task] - deadline(task_info), f"tardiness_{task}")

    model.addConstr(total_distance == quicksum(distance_terms))
    model.addConstr(total_travel_time == quicksum(travel_time_terms))
    model.setObjective(
        Config.WEIGHT * total_travel_time + (1 - Config.WEIGHT) * quicksum(tardiness[i] for i in tasks),
        GRB.MINIMIZE,
    )

    model.setParam("MIPFocus", 1)
    model.setParam("Heuristics", 0.5)
    model.setParam("NoRelHeurTime", 300)
    model.setParam("ImproveStartTime", 600)
    model.setParam("SubMIPNodes", 2000)
    model.setParam("RINS", 10)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", gurobi_threads)
    model.setParam("OutputFlag", 1 if verbose else 0)
    model.optimize()

    result = {
        "instance": instance_name,
        "status": model.Status,
        "status_name": _status_name(model.Status),
        "runtime": model.Runtime,
        "sol_count": model.SolCount,
        "best_bound": getattr(model, "ObjBound", None),
        "mip_gap": getattr(model, "MIPGap", None) if model.SolCount > 0 else None,
        "solver_objective": model.ObjVal if model.SolCount > 0 else None,
    }
    if model.SolCount == 0:
        result.update({"fitness": " ", "distance": " ", "travel_time": " ", "tardiness": " ", "solution": " "})
        results[instance_name] = result
        if verbose:
            print(_format_solution_report(result))
        return None

    path_map = _build_path_map(x, instance)
    sequence_map, path_init_task_map = path_map2sequence_map(path_map)
    solution = Solution(instance, sequence_map, path_init_task_map)
    fitness = solution.get_fitness()

    result.update({
        "fitness": fitness,
        "distance": solution.distance,
        "travel_time": solution.travel_time,
        "tardiness": solution.tardiness,
        "solution": str(solution.get_path_map()),
    })
    results[instance_name] = result
    if verbose:
        print(_format_solution_report(result))
    return solution


if __name__ == "__main__":
    build_model("RW_N20_K3_M16_I6.xlsx", time_limit=3600, verbose=True)
