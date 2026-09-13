import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import statistics
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from package_common import (  # noqa: E402
    EVAL_SEED, KAPPAS, METHODS, MODES, PACKAGE_ROOT, ROBOT_NUM_LISTS,
    SAMPLE_WIDTHS, SIZES, TIMING_REPEATS, cell_name, load_json, method_slug,
    numeric_instance_index, package_relative, task_id,
    validate_run_id, write_json,
)
from replay_metrics import replay_schedule  # noqa: E402


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def set_eval_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_method_runtime(method_dir):
    sys.path.insert(0, str(method_dir.resolve()))
    from utils import load_model, move_to
    return load_model, move_to


def decode_one(model, cpu_batch, device, width, move_to):
    batch = move_to(cpu_batch, device)
    model.set_decode_type("greedy" if width == 0 else "sampling", temp=1.0)
    with torch.no_grad():
        sequence, cost = model.sample_many(
            batch, batch_rep=1 if width == 0 else width, iter_rep=1
        )
    if sequence is None or cost is None or len(cost) != 1:
        raise RuntimeError("Expected one feasible solution for one instance")
    return sequence[0], cost[0]


def query_gpu_contention(physical_gpu_id, allowed_pids=()):
    command = [
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu = subprocess.run([
            "nvidia-smi", "--id={}".format(physical_gpu_id),
            "--query-gpu=uuid", "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True, timeout=20)
        uuids = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
        if len(uuids) != 1:
            raise OSError("Unable to identify the requested GPU")
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as error:
        return {"checked_at": utc_now(), "available": False, "contended": True, "error": str(error), "foreign_processes": []}
    if completed.returncode != 0:
        return {
            "checked_at": utc_now(), "available": False, "contended": True,
            "error": completed.stderr.strip() or "nvidia-smi failed", "foreign_processes": [],
        }
    foreign = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] == uuids[0] and parts[1].isdigit() and int(parts[1]) not in {os.getpid(), *allowed_pids}:
            foreign.append({"pid": int(parts[1]), "used_gpu_memory_mib": parts[2] if len(parts) > 2 else "unknown"})
    return {
        "checked_at": utc_now(), "available": True, "contended": bool(foreign),
        "error": None, "foreign_processes": foreign,
    }


def atomic_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_pickle(path, payload):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def instance_files(kappa, size, count=100):
    cell = cell_name(kappa, size)
    cell_dir = PACKAGE_ROOT / "instances" / "synthetic" / "type_{}".format(kappa) / "Instance" / cell
    files = sorted(cell_dir.glob("*.xlsx"), key=numeric_instance_index)
    if len(files) != 100 or [numeric_instance_index(path) for path in files] != list(range(1, 101)):
        raise ValueError("Instance cell is not exactly I1-I100: {}".format(cell))
    if not 1 <= count <= 100 or any(path.name != f'{cell}_I{numeric_instance_index(path)}.xlsx' for path in files):
        raise ValueError('Invalid instance count or scale in filename')
    sys.path.insert(0, str(PACKAGE_ROOT / 'scripts'))
    from generate_synthetic_extension import read_instance, validate_rows
    for path in files[:count]:
        validate_rows(read_instance(path), n=size, kappa=kappa)
    return files[:count]


def load_instances(model, kappa, size, count):
    files = instance_files(kappa, size, count)
    os.environ["COHET_INSTANCE_DIR"] = str(files[0].parent)
    items = []
    for path in files:
        dataset = model.problem.make_dataset(filename=path.name, num_samples=1, offset=0)
        if len(dataset) != 1:
            raise ValueError("Failed to load exactly one instance: {}".format(path.name))
        items.append((path, dataset[0]))
    return items


def audit_solution(cpu_item, sequence, model_cost, robot_num_list):
    tour = sequence.detach().cpu().reshape(-1).tolist()
    result = replay_schedule(
        source=cpu_item["source"].cpu().numpy(),
        deadline=cpu_item["deadline"].cpu().numpy(),
        operation_time=cpu_item["operation_time"].cpu().numpy(),
        tour=tour,
        robot_num_list=robot_num_list,
        velocity=1.0,
        distance_weight=0.5,
    )
    model_objective = float(model_cost.detach().cpu().item())
    model_residual = abs(model_objective - result["objective"])
    scale = np.float32(max(abs(model_objective), abs(result["objective"])))
    # Batched GPU reductions can differ from the serial float32 replay by a
    # few representable values. The replay remains the canonical result.
    model_tolerance = max(1e-6, 4.0 * abs(float(np.spacing(scale))))
    if model_residual > model_tolerance:
        raise ValueError("Model/replay objective mismatch: {:.12g}".format(model_residual))
    objective = result["objective"]
    objective_error = abs(objective - result["objective"])
    return tour, objective, result, objective_error, model_objective, model_residual, model_tolerance


def quality_phase(model, instances, device, width, move_to, identity, output_dir):
    rows = []
    raw = []
    for path, cpu_item in instances:
        sequence, cost = decode_one(model, default_collate([cpu_item]), device, width, move_to)
        tour, objective, replay, error, model_objective, model_residual, model_tolerance = audit_solution(
            cpu_item, sequence, cost, json.loads(identity["robot_num_list"])
        )
        index = numeric_instance_index(path)
        row = {
            **identity,
            "sample_width": width,
            "training_seed": 1234,
            "eval_seed": EVAL_SEED,
            "instance_index": index,
            "instance_id": "I{}".format(index),
            "instance_filename": path.name,
            "objective": objective,
            "distance": replay["distance"],
            "tardiness": replay["tardiness"],
            "replay_objective": replay["objective"],
            "objective_replay_error": error,
            "model_objective": model_objective,
            "model_replay_residual": model_residual,
            "model_replay_tolerance": model_tolerance,
            "feasible": True,
            "status": "SUCCEEDED",
            "tour": json.dumps(tour, separators=(",", ":")),
        }
        rows.append(row)
        raw.append({**row, "tour": tour})
    atomic_csv(output_dir / "per_instance.csv", rows)
    atomic_pickle(output_dir / "raw.pkl", raw)
    return {
        "row_count": len(rows),
        "max_objective_replay_error": max(row["objective_replay_error"] for row in rows),
        "max_model_replay_residual": max(row["model_replay_residual"] for row in rows),
        "contended": False,
    }


def timing_command(method_dir, checkpoint, cell, mode, output_pickle):
    return [
        sys.executable, "-B", "-u", str(method_dir / "eval.py"), cell,
        "--model", str(checkpoint.parent),
        "--decode_strategy", "sample" if mode == "sample1280" else "greedy",
        "--width", str(SAMPLE_WIDTHS[mode]), "--eval_batch_size", "1",
        "-o", str(output_pickle),
    ]


def read_timing_run(output_pickle, stdout_path):
    # Only read the trusted result produced by the local evaluator subprocess.
    with output_pickle.open("rb") as stream:
        results, parallelism = pickle.load(stream)
    if len(results) != 100 or parallelism != 1:
        raise ValueError("Expected 100 results with evaluation batch size 1")
    durations = [float(row[2]) for row in results]
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("Timing durations must be finite and positive")
    match = re.search(r"Average serial duration:\s*([0-9eE+.-]+)",
                      stdout_path.read_text(encoding="utf-8", errors="replace"))
    run_mean = statistics.fmean(durations)
    if match is None or abs(float(match.group(1)) - run_mean) > 1e-9:
        raise ValueError("Saved durations do not match the evaluator's logged mean")
    return results, run_mean


def timing_phase(method_dir, checkpoint, physical_gpu_id, identity, output_dir):
    files = instance_files(identity["kappa"], identity["size"])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(physical_gpu_id),
           "COHET_INSTANCE_DIR": str(files[0].parent.parent),
           "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    rows, run_means = [], []
    for repeat in range(1, TIMING_REPEATS + 1):
        before = query_gpu_contention(physical_gpu_id)
        if before["contended"]:
            raise RuntimeError("Timing requires an idle GPU and an available contention check")
        output_pickle = output_dir / f"repeat_{repeat}.pkl"
        stdout_path = output_dir / f"repeat_{repeat}.log"
        command = timing_command(method_dir, checkpoint, files[0].parent.name,
                                 identity["mode"], output_pickle)
        contended = False
        with stdout_path.open("xb") as stdout:
            process = subprocess.Popen(command, cwd=method_dir, env=env,
                                       stdout=stdout, stderr=subprocess.STDOUT)
            try:
                while True:
                    snapshot = query_gpu_contention(physical_gpu_id, (process.pid,))
                    contended = contended or snapshot["contended"]
                    if process.poll() is not None:
                        break
                    time.sleep(2)
                returncode = process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                raise
        if returncode != 0:
            raise RuntimeError(f"Original evaluator failed; see {stdout_path}")
        results, run_mean = read_timing_run(output_pickle, stdout_path)
        for position, (cost, _, duration) in enumerate(results, 1):
            rows.append({
                **{key: identity[key] for key in ("method", "kappa", "size", "mode")},
                "repeat": repeat, "position": position, "cost": float(cost),
                "duration_s": float(duration), "contended": contended,
            })
        atomic_csv(output_dir / "timing_raw.csv", rows)
        if contended:
            raise RuntimeError("GPU contention detected; this attempt is not valid timing")
        output_pickle.unlink()
        run_means.append(run_mean)
        print(f"Timing repeat {repeat}/{TIMING_REPEATS}: {run_mean:.9g} s/instance", flush=True)
    return {"row_count": len(rows), "run_means_s": run_means,
            "median_s": statistics.median(run_means), "contended": False}


def input_signature(method, kappa, size, checkpoint, args_path):
    if not checkpoint.is_file() or not args_path.is_file():
        raise FileNotFoundError("Checkpoint and adjacent args.json are required")
    return {"method": method, "kappa": kappa, "size": size,
            "checkpoint": package_relative(checkpoint), "model_options": load_json(args_path),
            "experiment": load_json(SCRIPT_DIR / "experiment.json")}


def parse_args():
    parser = argparse.ArgumentParser(description="Run one DRL main-table task")
    parser.add_argument("--phase", choices=("smoke", "quality", "timing"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--kappa", type=int, choices=KAPPAS, required=True)
    parser.add_argument("--size", type=int, choices=SIZES, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    options = parse_args()
    validate_run_id(options.run_id)
    if options.gpu_id < 0:
        raise ValueError("gpu-id must be non-negative")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(options.gpu_id))
    attempt_dir = options.attempt_dir.resolve()
    attempt_dir.relative_to(PACKAGE_ROOT)
    attempt_dir.mkdir(parents=True, exist_ok=False)

    method_dir = PACKAGE_ROOT / "methods" / "learning" / method_slug(options.method) / "type_{}".format(options.kappa)
    checkpoint = PACKAGE_ROOT / "checkpoints" / method_slug(options.method) / "type_{}".format(options.kappa) / "size_{}".format(options.size) / "epoch-99.pt"
    args_path = checkpoint.with_name("args.json")
    signature = input_signature(options.method, options.kappa, options.size, checkpoint, args_path)
    metadata = {
        "status": "RUNNING", "started_at": utc_now(), "phase": options.phase,
        "run_id": options.run_id, "method": options.method, "kappa": options.kappa,
        "size": options.size, "mode": options.mode, "sample_width": SAMPLE_WIDTHS[options.mode],
        "eval_seed": None if options.phase == "timing" else EVAL_SEED,
        "gpu_id": options.gpu_id, "input_signature": signature,
        "log_file": package_relative(os.environ["COHET_LOG_FILE"]) if os.environ.get("COHET_LOG_FILE") else None,
    }
    write_json(attempt_dir / "metadata.json", metadata)
    (attempt_dir / "command.txt").write_text(
        "cwd={}\nCUDA_VISIBLE_DEVICES={}\ncommand={}\n".format(
            Path.cwd(), os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            shlex.join(["conda", "run", "--no-capture-output", "-n", "my310env", "python", "-u", *sys.argv]),
        ), encoding="utf-8",
    )
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(options.gpu_id):
            raise RuntimeError("CUDA_VISIBLE_DEVICES must contain exactly the requested physical GPU id")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; GPU evaluation is required")
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        model_args = signature["model_options"]
        if int(model_args["graph_size"]) != options.size or int(model_args["seed"]) != 1234:
            raise ValueError("Checkpoint args do not match size and seed contract")
        identity = {
            "method": options.method, "kappa": options.kappa, "size": options.size,
            "mode": options.mode, "robot_num_list": json.dumps(ROBOT_NUM_LISTS[options.kappa]),
        }
        width = SAMPLE_WIDTHS[options.mode]
        if options.phase in ("smoke", "quality"):
            set_eval_seed(EVAL_SEED)
            load_model, move_to = import_method_runtime(method_dir)
            model, _ = load_model(str(checkpoint))
            device = torch.device("cuda:0")
            model.to(device)
            model.eval()
            count = 1 if options.phase == "smoke" else 100
            instances = load_instances(model, options.kappa, options.size, count)
            result = quality_phase(model, instances, device, width, move_to, identity, attempt_dir)
        else:
            result = timing_phase(method_dir, checkpoint, options.gpu_id, identity, attempt_dir)
        metadata.update({"status": "SUCCEEDED", "finished_at": utc_now(), "result": result})
        write_json(attempt_dir / "metadata.json", metadata)
        status_path = PACKAGE_ROOT / "results" / options.run_id / "status" / options.phase / "{}.json".format(task_id(options.method, options.kappa, options.size, options.mode))
        write_json(status_path, {
            "status": "SUCCEEDED", "attempt_dir": package_relative(attempt_dir),
        })
        print("SUCCEEDED phase={} task={} rows={}".format(options.phase, task_id(options.method, options.kappa, options.size, options.mode), result["row_count"]), flush=True)
    except BaseException as error:
        metadata.update({"status": "FAILED", "finished_at": utc_now(), "error_type": type(error).__name__, "error": str(error)})
        write_json(attempt_dir / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
