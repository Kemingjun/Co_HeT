from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from method_registry import REPO_ROOT, default_dataset_name, get_instance_root, get_method_dir, infer_size_from_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Co-HeT and DRL baselines with a unified interface.")
    parser.add_argument("--method", required=True, choices=["cohet", "am", "hdrl", "tdrl", "mvmoe", "echo"])
    parser.add_argument("--robot_type", type=int, choices=[2, 3], default=2)
    parser.add_argument("--dataset", default="Synthetic_Dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--decode_strategy", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--val_size", type=int, default=10000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--results_dir", default=str(REPO_ROOT / "results" / "eval"))
    parser.add_argument("--real_world", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_progress_bar", action="store_true")
    parser.add_argument("-f", "--force", action="store_true")
    return parser.parse_args()


def resolve_dataset_name(args: argparse.Namespace) -> str:
    if args.dataset in {"Synthetic_Dataset", "Real_World_Dataset"}:
        size = infer_size_from_model(args.model)
        return default_dataset_name(args.robot_type, size, real_world=args.real_world)
    return args.dataset


def main() -> int:
    args = parse_args()
    method_dir = get_method_dir(args.method, args.robot_type, real_world=args.real_world)
    dataset_name = resolve_dataset_name(args)
    model_path = Path(args.model).resolve()
    instance_root = get_instance_root(args.robot_type, real_world=args.real_world)

    command = [
        sys.executable,
        "eval.py",
        dataset_name,
        "--model",
        str(model_path),
        "--decode_strategy",
        args.decode_strategy,
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--val_size",
        str(args.val_size),
        "--offset",
        str(args.offset),
        "--results_dir",
        str(Path(args.results_dir).resolve()),
    ]
    if args.decode_strategy == "sample":
        width = args.width or 1280
        command.extend(["--width", str(width)])
    if args.no_cuda:
        command.append("--no_cuda")
    if args.no_progress_bar:
        command.append("--no_progress_bar")
    if args.force:
        command.append("-f")

    env = os.environ.copy()
    env["COHET_INSTANCE_DIR"] = str(instance_root)
    print("Running:", " ".join(command))
    print("Method directory:", method_dir)
    print("Instance root:", instance_root)
    return subprocess.call(command, cwd=method_dir, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
