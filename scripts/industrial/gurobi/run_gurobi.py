from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from run_support import (
    DATASET_ID,
    PACKAGE_ROOT,
    load_experiment_config,
    select_instances,
    write_json_once,
    write_or_validate_config,
)


def _run_one(task: dict) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "methods/conventional/industrial/gurobi"))
    os.environ["COHET_INDUSTRIAL_INSTANCE_DIR"] = str(Path(__file__).resolve().parents[3] / "instances/real_world_test100_seed20260906")
    started = time.perf_counter()
    record = {**task}
    try:
        from GUROBI.GUROBI import build_model

        solver_rows = {}
        solution = build_model(
            task["instance"],
            results=solver_rows,
            time_limit=task["time_limit_s"],
            verbose=False,
            gurobi_threads=task["gurobi_threads"],
        )
        solver = solver_rows[task["instance"]]
        record.update({
            "runtime_s": float(solver.get("runtime", time.perf_counter() - started)),
            "solver_status": solver.get("status"),
            "solver_status_name": solver.get("status_name"),
            "solution_count": solver.get("sol_count"),
            "incumbent": solver.get("solver_objective"),
            "solver_objective": solver.get("solver_objective"),
            "best_bound": solver.get("best_bound"),
            "mip_gap": solver.get("mip_gap"),
            "status": "ok" if solution is not None else "no_incumbent",
            "error": "",
        })
        if solution is not None:
            objective = float(solution.get_fitness())
            solver_objective = float(solver["solver_objective"])
            objective_from_components = (
                0.5 * float(solution.travel_time) + 0.5 * float(solution.tardiness)
            )
            solver_replay_abs_error = abs(solver_objective - objective)
            replay_formula_abs_error = abs(objective - objective_from_components)
            objective_replay_error = max(solver_replay_abs_error, replay_formula_abs_error)
            record.update(
                {
                    "travel_distance": float(solution.distance),
                    "travel_time": float(solution.travel_time),
                    "tardiness": float(solution.tardiness),
                    "objective": objective,
                    "objective_from_components": objective_from_components,
                    "solver_replay_abs_error": solver_replay_abs_error,
                    "replay_formula_abs_error": replay_formula_abs_error,
                    "objective_replay_error": objective_replay_error,
                    "objective_consistent": objective_replay_error <= 1e-6,
                    "path_map": {str(key): value for key, value in solution.get_path_map().items()},
                    "task_timeline": {str(key): value for key, value in solution.info_map.items()},
                }
            )
            if not record["objective_consistent"]:
                print(
                    f"WARNING: {task['instance']} objective replay error "
                    f"{objective_replay_error:.12g} exceeds 1e-6 "
                    f"(solver_replay={solver_replay_abs_error:.12g}, "
                    f"replay_formula={replay_formula_abs_error:.12g})",
                    flush=True,
                )
        return record
    except Exception as exc:
        record.setdefault("runtime_s", time.perf_counter() - started)
        record.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        )
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Run industrial Gurobi under the frozen travel-time objective.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--instance-start", type=int, default=1)
    parser.add_argument("--instance-end", type=int, default=100)
    parser.add_argument("--time-limit-s", type=float, default=3600)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--gurobi-threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."} or "\\" in args.run_id:
        raise ValueError("run-id must be a directory name, not a path")
    if not args.diagnostic and args.time_limit_s != 3600:
        raise ValueError("Formal Gurobi time limit is locked to 3600 seconds")
    if args.max_workers < 1:
        raise ValueError("max-workers must be a positive integer")
    if args.gurobi_threads < 1:
        raise ValueError("gurobi-threads must be a positive integer")

    instances = select_instances(
        sizes=args.sizes, instance_start=args.instance_start, instance_end=args.instance_end
    )
    run_config = {
        "dataset_id": DATASET_ID,
        "kind": "gurobi_diagnostic" if args.diagnostic else "gurobi",
        "sizes": sorted(set(args.sizes)),
        "instance_start": args.instance_start,
        "instance_end": args.instance_end,
        "time_limit_s": args.time_limit_s,
        "max_workers": args.max_workers,
        "gurobi_threads": args.gurobi_threads,
        "experiment": load_experiment_config(),
    }
    print(
        f"Gurobi tasks={len(instances)} workers={args.max_workers} "
        f"threads={args.gurobi_threads}"
    )
    if args.dry_run:
        return 0

    run_root = PACKAGE_ROOT / ("diagnostics" if args.diagnostic else "results/raw/gurobi") / args.run_id
    if run_root.exists() and not args.resume:
        raise FileExistsError("Run directory exists; use --resume or a new run-id")
    write_or_validate_config(run_root / "config.json", run_config)
    pending = []
    for instance_path in instances:
        size = int(instance_path.stem.split("_N", 1)[1].split("_", 1)[0])
        index = int(instance_path.stem.rsplit("_I", 1)[1])
        output = run_root / f"n{size}" / f"I{index}.json"
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if args.resume and existing.get("status") == "ok":
                continue
            raise FileExistsError(f"Raw result already exists: {output}")
        pending.append(
            {
                "method": "Gurobi",
                "dataset_id": DATASET_ID,
                "size": size,
                "instance": instance_path.name,
                "instance_index": index,
                "time_limit_s": args.time_limit_s,
                "gurobi_threads": args.gurobi_threads,
                "output_path": str(output),
            }
        )

    failures = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_run_one, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    **task,
                    "status": "failed",
                    "error": f"WorkerProcessError: {type(exc).__name__}: {exc}",
                }
            output = Path(record.pop("output_path"))
            if record.get("status") == "failed":
                failures += 1
            write_json_once(output, record)
            print(f"{record['instance']}: {record['status']}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
