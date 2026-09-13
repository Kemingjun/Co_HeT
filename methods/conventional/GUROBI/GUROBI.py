"""Co-HeT Gurobi model with package-local runtime configuration.

Index 0 represents two zero-cost virtual nodes sharing one index: arcs 0->task
leave the virtual source, arcs task->0 enter the virtual sink, and x[0,0,r]
represents an unused robot. No physical return-to-depot distance or time is
included. Constraints c0-c14 define the formulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from instance_io import Instance
from GUROBI.model_config import BIG_M, OBJECTIVE_TOLERANCE, RobotConfig, robot_config


@dataclass
class ModelArtifacts:
    model: gp.Model
    instance: Instance
    config: RobotConfig
    x: gp.tupledict
    arrival: gp.tupledict
    completion: gp.tupledict
    start: gp.tupledict
    finish: gp.tupledict
    task_tardiness: gp.tupledict
    distance_var: gp.Var
    tardiness_var: gp.Var
    distance_map: dict[tuple[int, int], float]
    big_m: float


def manhattan(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _position(instance: Instance, index: int) -> tuple[float, float]:
    if index == 0:
        return (0.0, 0.0)
    task = instance.tasks[index - 1]
    return (task.x, task.y)


def build_model(
    instance: Instance,
    *,
    time_limit_s: int,
    mip_gap: float,
    threads: int,
    seed: int,
    log_file: Path,
    log_to_console: bool = False,
) -> ModelArtifacts:
    if time_limit_s <= 0 or threads <= 0 or mip_gap < 0:
        raise ValueError("Invalid Gurobi limits")
    config = robot_config(instance.kappa)
    task_num = instance.n
    distance_map = {
        (i, j): manhattan(_position(instance, i), _position(instance, j))
        for i in range(task_num + 1)
        for j in range(task_num + 1)
    }

    model = gp.Model(f"cohet_k{instance.kappa}_n{instance.n}")
    x = model.addVars(
        (
            (i, j, r)
            for i in range(task_num + 1)
            for j in range(task_num + 1)
            for r in range(1, config.robot_num + 1)
        ),
        vtype=GRB.BINARY,
        name="x",
    )
    time_keys = (
        (i, k)
        for i in range(task_num + 1)
        for k in range(1, instance.kappa + 1)
    )
    arrival = model.addVars(list(time_keys), vtype=GRB.CONTINUOUS, name="A")
    completion = model.addVars(
        (
            (i, k)
            for i in range(task_num + 1)
            for k in range(1, instance.kappa + 1)
        ),
        vtype=GRB.CONTINUOUS,
        name="C",
    )
    start = model.addVars(range(task_num + 1), vtype=GRB.CONTINUOUS, name="S")
    finish = model.addVars(range(task_num + 1), vtype=GRB.CONTINUOUS, name="F")
    task_tardiness = model.addVars(
        range(task_num + 1), vtype=GRB.CONTINUOUS, name="T"
    )
    distance_var = model.addVar(vtype=GRB.CONTINUOUS, name="distance")
    tardiness_var = model.addVar(vtype=GRB.CONTINUOUS, name="tardiness")

    model.addConstr(
        distance_var
        == gp.quicksum(
            x[i, j, r] * distance_map[i, j]
            for i in range(task_num + 1)
            for j in range(1, task_num + 1)
            for r in range(1, config.robot_num + 1)
        ),
        "distance_total",
    )
    model.addConstr(
        tardiness_var == gp.quicksum(task_tardiness[i] for i in range(1, task_num + 1)),
        "tardiness_total",
    )
    model.setObjective(0.5 * distance_var + 0.5 * tardiness_var, GRB.MINIMIZE)

    # Constraints c0-c14.
    for i in range(1, task_num + 1):
        for r in range(1, config.robot_num + 1):
            model.addConstr(x[i, i, r] == 0, f"c0_{i}_{i}_{r}")

    for k, (robot_start, robot_end) in enumerate(config.type_list, start=1):
        for j in range(1, task_num + 1):
            model.addConstr(
                gp.quicksum(
                    x[i, j, r]
                    for i in range(task_num + 1)
                    for r in range(robot_start, robot_end)
                )
                == 1,
                f"c1_{j}_{k}",
            )
        for i in range(1, task_num + 1):
            model.addConstr(
                gp.quicksum(
                    x[i, j, r]
                    for j in range(task_num + 1)
                    for r in range(robot_start, robot_end)
                )
                == 1,
                f"c2_{i}_{k}",
            )

    for j in range(1, task_num + 1):
        for r in range(1, config.robot_num + 1):
            model.addConstr(
                gp.quicksum(x[i, j, r] for i in range(task_num + 1))
                - gp.quicksum(x[j, i, r] for i in range(task_num + 1))
                == 0,
                f"c3_{j}_{r}",
            )

    for r in range(1, config.robot_num + 1):
        model.addConstr(
            gp.quicksum(x[0, i, r] for i in range(task_num + 1)) == 1,
            f"c5_{r}",
        )
        model.addConstr(
            gp.quicksum(x[i, 0, r] for i in range(task_num + 1)) == 1,
            f"c6_{r}",
        )

    for k in range(1, instance.kappa + 1):
        model.addConstr(completion[0, k] == 0, f"c7_{k}")

    for k, (robot_start, robot_end) in enumerate(config.type_list, start=1):
        for r in range(robot_start, robot_end):
            for j in range(1, task_num + 1):
                for i in range(task_num + 1):
                    model.addConstr(
                        completion[i, k]
                        + distance_map[i, j]
                        - BIG_M * (1 - x[i, j, r])
                        <= arrival[j, k],
                        f"c8_{k}_{r}_{j}_{i}",
                    )

    for k in range(1, instance.kappa + 1):
        for i in range(1, task_num + 1):
            model.addConstr(arrival[i, k] <= start[i], f"c10_{k}_{i}")
            model.addConstr(
                start[i] + instance.tasks[i - 1].processing_times[k - 1]
                <= completion[i, k],
                f"c11_{k}_{i}",
            )
            model.addConstr(completion[i, k] <= finish[i], f"c12_{k}_{i}")

    for i in range(1, task_num + 1):
        model.addConstr(
            finish[i] - instance.tasks[i - 1].deadline <= task_tardiness[i],
            f"c13_{i}",
        )
        model.addConstr(task_tardiness[i] >= 0, f"c14_{i}")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    if task_num == 20:
        model.setParam("MIPFocus", 1)
        model.setParam("Heuristics", 0.10)
        model.setParam("NoRelHeurWork", 60)
    elif task_num == 50:
        model.setParam("MIPFocus", 1)
        model.setParam("Heuristics", 0.20)
        model.setParam("NoRelHeurWork", 240)
    model.setParam("TimeLimit", float(time_limit_s))
    model.setParam("MIPGap", float(mip_gap))
    model.setParam("Threads", int(threads))
    model.setParam("Seed", int(seed))
    model.setParam("LogFile", str(log_file.resolve()))
    model.setParam("LogToConsole", 1 if log_to_console else 0)
    return ModelArtifacts(
        model=model,
        instance=instance,
        config=config,
        x=x,
        arrival=arrival,
        completion=completion,
        start=start,
        finish=finish,
        task_tardiness=task_tardiness,
        distance_var=distance_var,
        tardiness_var=tardiness_var,
        distance_map=distance_map,
        big_m=BIG_M,
    )


def classify_status(model: gp.Model) -> str:
    status = int(model.Status)
    has_solution = int(model.SolCount) > 0
    if status == GRB.OPTIMAL:
        return "optimal"
    if status == GRB.TIME_LIMIT:
        return "time_limit_feasible" if has_solution else "time_limit_no_incumbent"
    if status == GRB.INFEASIBLE:
        return "infeasible"
    if status == GRB.INF_OR_UNBD:
        return "inf_or_unbd"
    if status == GRB.UNBOUNDED:
        return "unbounded"
    if status == GRB.SUBOPTIMAL and has_solution:
        return "suboptimal_feasible"
    return "solver_limit_feasible" if has_solution else "solver_limit_no_incumbent"


def _robot_type(config: RobotConfig, robot: int) -> int:
    for k, (start, end) in enumerate(config.type_list, start=1):
        if start <= robot < end:
            return k
    raise ValueError(f"Robot {robot} is outside the configured ranges")


def extract_path_map(artifacts: ModelArtifacts) -> dict[int, list[int]]:
    n = artifacts.instance.n
    paths: dict[int, list[int]] = {}
    for robot in range(1, artifacts.config.robot_num + 1):
        connection: dict[int, int] = {}
        selected = 0
        for i in range(n + 1):
            for j in range(n + 1):
                if artifacts.x[i, j, robot].X > 0.5:
                    if i in connection:
                        raise ValueError(f"Robot {robot} has multiple outgoing arcs from {i}")
                    connection[i] = j
                    selected += 1
        if connection.get(0) == 0:
            if selected != 1:
                raise ValueError(f"Unused robot {robot} has non-dummy arcs")
            paths[robot] = []
            continue
        current = connection.get(0)
        path: list[int] = []
        seen: set[int] = set()
        while current != 0:
            if current is None or current in seen:
                raise ValueError(f"Robot {robot} path is disconnected or cyclic")
            seen.add(current)
            path.append(current)
            current = connection.get(current)
        if selected != len(path) + 1:
            raise ValueError(f"Robot {robot} contains disconnected selected arcs")
        paths[robot] = path
    return paths


def evaluate_paths(instance: Instance, path_map: dict[int, list[int]]) -> dict[str, Any]:
    config = robot_config(instance.kappa)
    predecessors: dict[tuple[int, int], int] = {}
    distance = 0.0
    for robot, path in path_map.items():
        k = _robot_type(config, robot)
        predecessor = 0
        for task_index in path:
            key = (task_index, k)
            if key in predecessors:
                raise ValueError(f"Task {task_index} is duplicated for type {k}")
            predecessors[key] = predecessor
            distance += manhattan(_position(instance, predecessor), _position(instance, task_index))
            predecessor = task_index
    expected = {(i, k) for i in range(1, instance.n + 1) for k in range(1, instance.kappa + 1)}
    if set(predecessors) != expected:
        raise ValueError("Paths do not cover every task exactly once for every robot type")

    remaining = set(range(1, instance.n + 1))
    starts: dict[int, float] = {}
    completions: dict[tuple[int, int], float] = {}
    while remaining:
        ready = [
            task_index
            for task_index in sorted(remaining)
            if all(
                predecessors[task_index, k] == 0
                or (predecessors[task_index, k], k) in completions
                for k in range(1, instance.kappa + 1)
            )
        ]
        if not ready:
            raise ValueError("Task precedence graph contains a cross-type cycle")
        for task_index in ready:
            arrival_times = []
            for k in range(1, instance.kappa + 1):
                predecessor = predecessors[task_index, k]
                previous_completion = 0.0 if predecessor == 0 else completions[predecessor, k]
                arrival_times.append(
                    previous_completion
                    + manhattan(_position(instance, predecessor), _position(instance, task_index))
                )
            start_time = max(arrival_times)
            starts[task_index] = start_time
            task = instance.tasks[task_index - 1]
            for k in range(1, instance.kappa + 1):
                completions[task_index, k] = start_time + task.processing_times[k - 1]
            remaining.remove(task_index)

    tardiness = 0.0
    task_times: dict[str, Any] = {}
    for task in instance.tasks:
        finish = max(completions[task.index, k] for k in range(1, instance.kappa + 1))
        task_tardiness = max(finish - task.deadline, 0.0)
        tardiness += task_tardiness
        task_times[str(task.index)] = {
            "start": starts[task.index],
            "completion_by_type": [
                completions[task.index, k] for k in range(1, instance.kappa + 1)
            ],
            "finish": finish,
            "tardiness": task_tardiness,
        }
    return {
        "distance": distance,
        "tardiness": tardiness,
        "objective": 0.5 * distance + 0.5 * tardiness,
        "task_times": task_times,
    }


def extract_solution(
    artifacts: ModelArtifacts, *, tolerance: float = OBJECTIVE_TOLERANCE
) -> dict[str, Any]:
    if artifacts.model.SolCount <= 0:
        raise ValueError("The model has no incumbent solution")
    path_map = extract_path_map(artifacts)
    independently_evaluated = evaluate_paths(artifacts.instance, path_map)
    objective = float(artifacts.model.ObjVal)
    distance = float(artifacts.distance_var.X)
    tardiness = float(artifacts.tardiness_var.X)
    errors = {
        "objective": abs(objective - independently_evaluated["objective"]),
        "distance": abs(distance - independently_evaluated["distance"]),
        "tardiness": abs(tardiness - independently_evaluated["tardiness"]),
    }
    consistent = max(errors.values()) <= tolerance
    if not consistent:
        raise ValueError(f"MILP and independent evaluation differ: {errors}")
    return {
        "objective": objective,
        "distance": distance,
        "tardiness": tardiness,
        "path_map": {str(robot): path for robot, path in path_map.items()},
        "task_times": independently_evaluated["task_times"],
        "independent_evaluation": independently_evaluated,
        "consistency_errors": errors,
        "objective_consistent": True,
    }


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None
