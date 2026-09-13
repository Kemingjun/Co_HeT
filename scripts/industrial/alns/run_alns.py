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
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "methods/conventional/industrial/alns"))
    os.environ["COHET_INDUSTRIAL_INSTANCE_DIR"] = str(Path(__file__).resolve().parents[3] / "instances/real_world_test100_seed20260906")
    from Algorithms.ALNS import ALNS

    started = time.perf_counter()
    try:
        solution = ALNS(
            Path(task["instance"]).stem,
            max_iterations=task["iteration_limit"],
            time_limit=task["time_limit_s"],
            seed=task["seed"],
            verbose=False,
        )
        objective = float(solution.get_fitness())
        replay = 0.5 * float(solution.travel_time) + 0.5 * float(solution.tardiness)
        replay_error = abs(objective - replay)
        if replay_error > 1e-6:
            raise RuntimeError(f"Objective replay error {replay_error:.12g} exceeds 1e-6")
        return {
            **task,
            "travel_distance": float(solution.distance),
            "travel_time": float(solution.travel_time),
            "tardiness": float(solution.tardiness),
            "objective": objective,
            "objective_replay_error": replay_error,
            "runtime_s": time.perf_counter() - started,
            "path_map": {str(key): value for key, value in solution.get_path_map().items()},
            "task_timeline": {str(key): value for key, value in solution.info_map.items()},
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        return {
            **task,
            "runtime_s": time.perf_counter() - started,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run industrial ALNS under the frozen travel-time objective.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--instance-start", type=int, default=1)
    parser.add_argument("--instance-end", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iteration-limit", type=int, default=100)
    parser.add_argument("--time-limit-s", type=float, default=3600)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260603)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.run_id or Path(args.run_id).name != args.run_id or any(c in args.run_id for c in "/\\:") or args.run_id in {".", ".."}:
        raise ValueError("run-id must be a single directory name")
    if min(args.repetitions, args.iteration_limit, args.max_workers) < 1 or not math.isfinite(args.time_limit_s) or args.time_limit_s <= 0:
        raise ValueError("Repetitions, iterations, workers, and time limit must be positive")
    if not args.diagnostic and (args.repetitions, args.iteration_limit, args.time_limit_s) != (5, 100, 3600):
        raise ValueError("Formal ALNS budget is locked to 5 repetitions, 100 iterations, and 3600 seconds")
    instances = select_instances(
        sizes=args.sizes, instance_start=args.instance_start, instance_end=args.instance_end
    )
    run_config = {
        "dataset_id": DATASET_ID,
        "kind": "alns_diagnostic" if args.diagnostic else "alns",
        "sizes": sorted(set(args.sizes)),
        "instance_start": args.instance_start,
        "instance_end": args.instance_end,
        "repetitions": args.repetitions,
        "iteration_limit": args.iteration_limit,
        "time_limit_s": args.time_limit_s,
        "base_seed": args.base_seed,
        "experiment": load_experiment_config(),
    }
    tasks = []
    for path in instances:
        size = int(path.stem.split("_N", 1)[1].split("_", 1)[0])
        index = int(path.stem.rsplit("_I", 1)[1])
        for repetition in range(1, args.repetitions + 1):
            tasks.append(
                {
                    "dataset_id": DATASET_ID,
                    "method": "ALNS",
                    "size": size,
                    "instance": path.name,
                    "instance_index": index,
                    "repetition": repetition,
                    "seed": args.base_seed + size * 100000 + index * 100 + repetition,
                    "iteration_limit": args.iteration_limit,
                    "time_limit_s": args.time_limit_s,
                }
            )
    print(f"ALNS tasks={len(tasks)} workers={args.max_workers} dataset={DATASET_ID}", flush=True)
    if args.dry_run:
        return 0

    run_root = PACKAGE_ROOT / ("diagnostics" if args.diagnostic else "results/raw/alns") / args.run_id
    if run_root.exists() and not args.resume:
        raise FileExistsError("Run directory exists; use --resume or a new run-id")
    write_or_validate_config(run_root / "config.json", run_config)
    pending = []
    for task in tasks:
        output = run_root / f"n{task['size']}" / f"I{task['instance_index']}_rep{task['repetition']}.json"
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if args.resume and existing.get("status") == "ok":
                continue
            raise FileExistsError(f"Raw result already exists: {output}")
        task["output_path"] = str(output)
        pending.append(task)

    failures = 0
    print(f"Pending={len(pending)} skipped={len(tasks) - len(pending)} output={run_root}", flush=True)
    if not pending:
        return 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_run_one, task): task for task in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            output = Path(result.pop("output_path"))
            if result.get("status") != "ok":
                failures += 1
            write_json_once(output, result)
            print(f"[{completed}/{len(pending)}] {result['instance']} rep={result['repetition']}: {result['status']}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
