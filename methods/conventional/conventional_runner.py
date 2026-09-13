"""Run synthetic MILP and metaheuristic baselines.

The metaheuristic loop uses count <= iteration_limit:
iteration_limit=100 can complete 101 outer iterations, with duration=3600.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

from instance_io import read_instance

REPO_ROOT = Path(__file__).resolve().parents[2]
SOLVERS = ("gurobi", "alns", "iga", "dabc", "diwo")
TERMINAL = {"success", "optimal", "time_limit_feasible", "time_limit_no_incumbent",
            "infeasible", "inf_or_unbd", "unbounded", "solver_limit_feasible",
            "solver_limit_no_incumbent", "suboptimal_feasible"}
INSTANCE_PATTERN = re.compile(r"N(10|20|50|100)_K([23])_M(12|18)_I([1-9][0-9]?|100)(?:\.xlsx)?")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", required=True, choices=SOLVERS)
    parser.add_argument("instance", nargs="?", help="Optional canonical instance name, e.g. N20_K2_M12_I1")
    parser.add_argument("--kappas", "--kappa", nargs="+", type=int, choices=(2, 3))
    parser.add_argument("--sizes", "--size", nargs="+", type=int, choices=(10, 20, 50, 100))
    parser.add_argument("--instance-start", type=int, default=1)
    parser.add_argument("--instance-end", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=1, help="Set explicitly to reproduce the recorded repetition count")
    parser.add_argument("--iteration-limit", type=int, default=100)
    parser.add_argument("--time-limit-s", type=int, default=3600)
    parser.add_argument("--external-timeout-s", type=int, help="Default: 5400 for metaheuristics, 4500 for Gurobi")
    parser.add_argument("--gurobi-threads", type=int, default=1)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0, help="Gurobi seed; metaheuristic seeds depend on method and repetition")
    parser.add_argument("--instances-root", type=Path, default=REPO_ROOT / "instances" / "synthetic")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "outputs" / "conventional")
    parser.add_argument("--run-id", help="New output directory name; defaults to a UTC timestamp")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostic", action="store_true", help="Label explicitly altered numerical budgets")
    parser.add_argument("--dry-run", action="store_true", help="Validate selected workbooks; no solving or writing")
    return parser


def plan(args):
    if args.instance:
        match = INSTANCE_PATTERN.fullmatch(args.instance)
        if not match:
            raise ValueError("Expected a canonical synthetic instance name")
        n, kappa, robots, index = map(int, match.groups())
        if robots != {2: 12, 3: 18}[kappa]:
            raise ValueError("Instance robot count does not match kappa")
        if args.kappas or args.sizes or (args.instance_start, args.instance_end) != (1, 100):
            raise ValueError("Do not combine an instance name with matrix selection options")
        kappas, sizes, first, last = [kappa], [n], index, index
    else:
        if not args.kappas or not args.sizes:
            raise ValueError("Provide an instance name or both --kappas and --sizes")
        kappas, sizes, first, last = args.kappas, args.sizes, args.instance_start, args.instance_end
    if len(set(kappas)) != len(kappas) or len(set(sizes)) != len(sizes):
        raise ValueError("Duplicate kappa or size values are not allowed")
    if not 1 <= first <= last <= 100:
        raise ValueError("Instance range must be within 1..100")
    if args.repetitions <= 0 or args.iteration_limit <= 0 or args.time_limit_s <= 0:
        raise ValueError("Repetitions and budgets must be positive")
    if args.gurobi_threads <= 0 or not math.isfinite(args.mip_gap) or args.mip_gap < 0 or not 0 <= args.seed <= 2_000_000_000:
        raise ValueError("Invalid Gurobi parameters")
    is_gurobi = args.solver == "gurobi"
    if is_gurobi and (100 in sizes or args.repetitions != 1):
        raise ValueError("The result-producing Gurobi package supports n=10,20,50 and one solve per instance")
    if not is_gurobi and (args.seed != 0 or args.gurobi_threads != 1 or args.mip_gap != 0):
        raise ValueError("Gurobi-only options cannot change metaheuristic configuration")
    timeout = args.external_timeout_s if args.external_timeout_s is not None else (4500 if is_gurobi else 5400)
    if timeout <= 0:
        raise ValueError("External timeout must be positive")
    canonical = (args.time_limit_s == 3600 and args.iteration_limit == 100
                 and args.seed == 0 and args.gurobi_threads == 1 and args.mip_gap == 0
                 and timeout == (4500 if is_gurobi else 5400))
    if not canonical and not args.diagnostic:
        raise ValueError("Noncanonical numerical parameters require --diagnostic")
    if args.resume and not args.run_id:
        raise ValueError("--resume requires an explicit --run-id")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) or run_id.endswith("."):
        raise ValueError("Invalid run-id")
    root = args.instances_root.resolve()
    jobs = []
    for kappa in sorted(kappas):
        for size in sorted(sizes):
            prefix = f"N{size}_K{kappa}_M{12 if kappa == 2 else 18}"
            directory = root / f"type_{kappa}" / "Instance" / prefix
            expected = {f"{prefix}_I{i}.xlsx" for i in range(1, 101)}
            actual = {path.name for path in directory.glob("*.xlsx")}
            if actual != expected:
                raise ValueError(f"Expected exactly canonical instances 1..100 in {directory}; missing={len(expected - actual)}, unexpected={len(actual - expected)}")
            for index in range(first, last + 1):
                workbook = directory / f"{prefix}_I{index}.xlsx"
                read_instance(workbook, n=size, kappa=kappa)
                for rep in range(args.repetitions):
                    jobs.append(dict(solver=args.solver, kappa=kappa, size=size, index=index,
                                     rep=rep, instance=workbook.stem, workbook=str(workbook),
                                     iteration_limit=args.iteration_limit, time_limit_s=args.time_limit_s,
                                     gurobi_threads=args.gurobi_threads, target_mip_gap=args.mip_gap, seed=args.seed,
                                     diagnostic=bool(args.diagnostic)))
    config = dict(solver=args.solver, kappas=sorted(kappas), sizes=sorted(sizes),
                  instance_start=first, instance_end=last, repetitions=args.repetitions,
                  iteration_limit=args.iteration_limit, time_limit_s=args.time_limit_s,
                  external_timeout_s=timeout, gurobi_threads=args.gurobi_threads,
                  mip_gap=args.mip_gap, seed=args.seed, diagnostic=bool(args.diagnostic),
                  instances_root=str(root), python_hash_seed="0",
                  seed_formula="202409 + hash((method.upper(), rep)) % 10000000",
                  metaheuristic_loop="count <= iteration_limit")
    return config, jobs, args.results_root.resolve() / run_id


def job_id(job):
    return f"k{job['kappa']}_n{job['size']}_i{job['index']:03d}_{job['solver']}_r{job['rep']:03d}"


def write_new(path, payload):
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def prepare_run(run_dir, config, *, resume):
    config_path = run_dir / "config.json"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"Run directory exists; use --resume: {run_dir}")
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing run configuration does not match this command")
    else:
        if resume:
            raise FileNotFoundError(f"Cannot resume missing run: {run_dir}")
        run_dir.mkdir(parents=True)
        write_new(config_path, config)


def execute_attempt(job, *, output, timeout_s):
    command = [sys.executable, "-B", str(Path(__file__).with_name("conventional_worker.py")),
               "--job", json.dumps(job), "--output", str(output)]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    output.parent.mkdir(parents=True, exist_ok=True)
    failure = None
    with output.with_suffix(".stdout.log").open("x", encoding="utf-8") as stdout, output.with_suffix(".stderr.log").open("x", encoding="utf-8") as stderr:
        try:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, env=env,
                                     cwd=Path(__file__).resolve().parent, timeout=timeout_s, check=False)
            if process.returncode != 0:
                failure = f"Worker exited with code {process.returncode}"
        except subprocess.TimeoutExpired:
            failure = "External worker timeout"
        except OSError as exc:
            failure = f"Worker launch failed: {exc}"
    if not output.is_file():
        write_new(output, {**job, "status": "timeout" if failure == "External worker timeout" else "failed",
                           "objective": None, "error": failure or "Worker produced no result"})
    record = json.loads(output.read_text(encoding="utf-8"))
    if failure and record.get("status") in TERMINAL:
        raise RuntimeError(f"{failure}; existing result retained at {output}")
    return record


def execute_run(config, jobs, run_dir, *, resume=False, attempt_executor=None):
    prepare_run(run_dir, config, resume=resume)
    executor = attempt_executor or execute_attempt
    failures = 0
    for job in jobs:
        directory = run_dir / "raw" / job_id(job)
        previous = []
        for path in directory.glob("attempt_*.json"):
            number = int(path.stem.split("_")[-1])
            record = json.loads(path.read_text(encoding="utf-8"))
            if any(record.get(key) != value for key, value in job.items()):
                raise ValueError(f"Existing result identity/configuration mismatch: {path}")
            previous.append((number, record))
        previous.sort(key=lambda item: item[0])
        terminal = [record for _, record in previous if record.get("status") in TERMINAL]
        if terminal:
            record = terminal[-1]
        else:
            number = max((i for i, _ in previous), default=0) + 1
            while any(directory.glob(f"attempt_{number:03d}.*")):
                number += 1
            output = directory / f"attempt_{number:03d}.json"
            try:
                record = executor(job, output=output, timeout_s=config["external_timeout_s"])
            except Exception as exc:
                if output.exists():
                    raise
                record = {**job, "status": "failed", "objective": None, "error": f"{type(exc).__name__}: {exc}"}
            if not output.exists():
                write_new(output, record)
        if record.get("objective") is None or record.get("status") not in TERMINAL or record.get("objective_consistent") is False:
            failures += 1
    print(f"output_dir={run_dir} jobs={len(jobs)} failed_or_no_incumbent_or_inconsistent={failures}")
    return 1 if failures else 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        config, jobs, run_dir = plan(args)
        if args.dry_run:
            print(json.dumps({"config": config, "planned_jobs": len(jobs), "first_instance": jobs[0]["instance"],
                              "last_instance": jobs[-1]["instance"], "output_dir": str(run_dir), "dry_run": True}, indent=2))
            return 0
        return execute_run(config, jobs, run_dir, resume=args.resume)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
