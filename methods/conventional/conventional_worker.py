"""One isolated solver invocation; numerical code follows the round-2 packages."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

from conventional_runner import write_new
from instance_io import read_instance


def load_solver(name):
    modules = {"alns": ("Algorithms.ALNS", "ALNS"), "iga": ("Algorithms.IG", "IGA"),
               "dabc": ("Algorithms.DABC", "DABC"), "diwo": ("Algorithms.DIWO", "DIWO")}
    module, function = modules[name]
    return getattr(importlib.import_module(module), function)


def solve_metaheuristic(job):
    import numpy as np
    from Util.Config import Config

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 must be set before the worker starts")
    Config.configure(job["kappa"])
    os.environ["COHET_INSTANCE_DIR"] = str(Path(job["workbook"]).parent)
    seed = 202409 + hash((job["solver"].upper(), job["rep"])) % 10_000_000
    random.seed(seed)
    np.random.seed(seed)
    solver = load_solver(job["solver"])
    solution, reported, runtime, iterations, reason = solver(
        job["instance"], iteration_limit=job["iteration_limit"], duration=job["time_limit_s"])
    objective = float(solution.get_fitness())
    distance, tardiness = float(solution.distance), float(solution.tardiness)
    consistent = all(math.isfinite(value) for value in (objective, distance, tardiness)) and abs(objective - 0.5 * (distance + tardiness)) <= 1e-6
    return dict(status="success" if consistent else "objective_inconsistent", rng_seed=seed,
                objective=objective, distance=distance, tardiness=tardiness,
                algorithm_reported_objective=float(reported), objective_consistent=consistent,
                runtime_s=float(runtime), iterations_completed=int(iterations), termination_reason=str(reason),
                path_map=solution.get_path_map(), solution_code=solution.code,
                error=None if consistent else "Solution objective does not match 0.5*distance+0.5*tardiness")


def solve_gurobi(job, log_file):
    import gurobipy as gp
    from gurobipy import GRB
    from GUROBI.GUROBI import build_model, classify_status, extract_solution, finite_or_none

    instance = read_instance(Path(job["workbook"]), n=job["size"], kappa=job["kappa"])
    artifacts = build_model(instance, time_limit_s=job["time_limit_s"], mip_gap=job["target_mip_gap"],
                            threads=job["gurobi_threads"], seed=job["seed"], log_file=log_file)
    model = artifacts.model

    def attribute(name):
        try:
            value = getattr(model, name)
        except (AttributeError, gp.GurobiError):
            return None
        return finite_or_none(value) if isinstance(value, float) else value

    try:
        model.optimize()
        status = classify_status(model)
        code, count = int(model.Status), int(model.SolCount)
        status_name = next((name for name in ("OPTIMAL", "TIME_LIMIT", "INFEASIBLE", "INF_OR_UNBD",
                          "UNBOUNDED", "SUBOPTIMAL", "INTERRUPTED", "NODE_LIMIT", "ITERATION_LIMIT",
                          "SOLUTION_LIMIT", "NUMERIC", "CUTOFF", "USER_OBJ_LIMIT", "WORK_LIMIT", "MEM_LIMIT")
                          if getattr(GRB, name, None) == code), f"STATUS_{code}")
        result = dict(status=status, solver_status_code=code, solver_status_name=status_name,
                      termination_reason=status, solution_count=count, has_incumbent=count > 0,
                      optimal=status == "optimal", objective=attribute("ObjVal") if count else None,
                      distance=float(artifacts.distance_var.X) if count else None,
                      tardiness=float(artifacts.tardiness_var.X) if count else None,
                      obj_bound=attribute("ObjBound"), obj_bound_c=attribute("ObjBoundC"),
                      mip_gap=attribute("MIPGap") if count else None, runtime_s=attribute("Runtime"),
                      work=attribute("Work"), node_count=attribute("NodeCount"),
                      objective_consistent=False, path_map=None, error=None,
                      big_m=artifacts.big_m, distance_metric="manhattan", objective_weights=[0.5, 0.5])
        if count:
            try:
                result.update(extract_solution(artifacts))
            except ValueError as exc:
                # Keep the incumbent/bound/status and disclose replay disagreement.
                result["error"] = str(exc)
                from GUROBI.GUROBI import extract_path_map, evaluate_paths
                try:
                    paths = extract_path_map(artifacts)
                    result["path_map"] = paths
                    result["independent_evaluation"] = evaluate_paths(instance, paths)
                except ValueError as replay_error:
                    result["replay_error"] = str(replay_error)
        return result
    finally:
        model.dispose()


def execute_job(job, *, log_file):
    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    # Validate schema before invoking either algorithm, including direct worker usage.
    read_instance(Path(job["workbook"]), n=job["size"], kappa=job["kappa"])
    result = solve_gurobi(job, log_file) if job["solver"] == "gurobi" else solve_metaheuristic(job)
    return {**job, **result, "method": job["solver"].upper(), "started_at": started,
            "ended_at": datetime.now(timezone.utc).isoformat(), "wall_runtime_s": time.perf_counter() - wall_start}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="One JSON job supplied by the publication runner")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"ERROR: Worker output already exists: {args.output}", file=sys.stderr)
        return 2
    job = json.loads(args.job)
    try:
        result = execute_job(job, log_file=args.output.with_suffix(".gurobi.log"))
        code = 0
    except Exception as exc:
        result = {**job, "status": "failed", "objective": None, "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc(), "ended_at": datetime.now(timezone.utc).isoformat()}
        code = 1
    write_new(args.output, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
