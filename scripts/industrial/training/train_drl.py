from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_support import PACKAGE_ROOT, environment_snapshot, load_experiment_config, write_json_once, write_or_validate_config


METHODS = ("cohet", "am", "mvmoe", "hdrl", "tdrl", "echo")


def build_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    frozen = load_experiment_config()["training"]
    smoke = bool(args.smoke)
    command = [
        sys.executable,
        "-u",
        "run.py",
        "--problem",
        "hrsp",
        "--graph_size",
        str(args.size),
        "--seed",
        str(args.seed),
        "--n_epochs",
        str(1 if smoke else frozen["epochs"]),
        "--batch_size",
        str(2 if smoke else frozen["batch_size"]),
        "--epoch_size",
        str(2 if smoke else frozen["epoch_size"]),
        "--val_size",
        str(2 if smoke else frozen["validation_size"]),
        "--checkpoint_epochs",
        str(1 if smoke else 10),
        "--run_name",
        args.run_id,
        "--output_dir",
        str((run_dir / "artifacts").resolve()),
        "--log_dir",
        str((run_dir / "tensorboard").resolve()),
        "--no_progress_bar",
    ]
    if args.method in {"cohet", "hdrl", "tdrl"}:
        command[5:5] = ["--robot_type_num", "3"]
    if smoke:
        command.extend(["--baseline", "exponential", "--no_tensorboard"])
    if args.device == "cpu":
        command.append("--no_cuda")
    if args.resume:
        command.extend(["--resume", str(Path(args.resume).resolve())])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one industrial DRL model under the frozen travel-time objective.")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--size", required=True, type=int, choices=[10, 20, 30, 40])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--resume", default=None, help="Checkpoint file from the same run configuration.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.seed != 1234:
        raise ValueError("The formal industrial training seed is locked to 1234")

    if not args.run_id or Path(args.run_id).name != args.run_id or any(c in args.run_id for c in "/\\:") or args.run_id in {".", ".."}:
        raise ValueError("run-id must be a single directory name")

    base_dir = PACKAGE_ROOT / ("diagnostics" if args.smoke else "runs/train")
    run_dir = base_dir / args.run_id
    run_config = {
        "kind": "drl_training_smoke" if args.smoke else "drl_training",
        "method": args.method,
        "size": args.size,
        "seed": args.seed,
        "device": args.device,
        "gpu_id": args.gpu_id,
        "experiment": load_experiment_config(),
    }
    command = build_command(args, run_dir)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    if run_dir.exists() and not args.resume:
        raise FileExistsError("Run directory exists; use --resume or a new run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_or_validate_config(run_dir / "config.json", run_config)
    attempt_dir = run_dir / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt_dir.mkdir(parents=True, exist_ok=False)
    write_json_once(
        attempt_dir / "launch.json",
        {
            "command": command,
            "environment": environment_snapshot(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    method_dir = Path(__file__).resolve().parents[3] / "methods/real_world" / args.method
    completed = subprocess.run(command, cwd=method_dir, env=env)
    write_json_once(
        attempt_dir / "status.json",
        {
            "returncode": completed.returncode,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "succeeded" if completed.returncode == 0 else "failed",
        },
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
