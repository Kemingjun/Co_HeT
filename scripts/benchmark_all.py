from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from method_registry import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch benchmark over DRL methods.")
    parser.add_argument("--dataset", default="Synthetic_Dataset")
    parser.add_argument("--methods", nargs="+", default=["cohet", "am", "hdrl", "tdrl", "mvmoe", "echo"])
    parser.add_argument("--robot_types", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--decode_strategies", nargs="+", default=["greedy", "sample"])
    parser.add_argument("--sample_width", type=int, default=1280)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--val_size", type=int, default=10000)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--out_prefix", default=str(REPO_ROOT / "results" / "comparison" / "benchmark"))
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_progress_bar", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for robot_type in args.robot_types:
        for method in args.methods:
            for size in args.sizes:
                model = REPO_ROOT / "checkpoints" / method / f"type_{robot_type}" / f"size_{size}"
                if not model.exists():
                    rows.append({"method": method, "robot_type": str(robot_type), "size": str(size), "status": "missing_model"})
                    continue
                for decode in args.decode_strategies:
                    command = [
                        sys.executable,
                        str(REPO_ROOT / "scripts" / "eval_drl.py"),
                        "--method",
                        method,
                        "--robot_type",
                        str(robot_type),
                        "--dataset",
                        args.dataset,
                        "--model",
                        str(model),
                        "--decode_strategy",
                        decode,
                        "--eval_batch_size",
                        str(args.eval_batch_size),
                        "--val_size",
                        str(args.val_size),
                        "--force",
                    ]
                    if decode == "sample":
                        command.extend(["--width", str(args.sample_width)])
                    if args.no_cuda:
                        command.append("--no_cuda")
                    if args.no_progress_bar:
                        command.append("--no_progress_bar")
                    status = subprocess.call(command)
                    rows.append(
                        {
                            "method": method,
                            "robot_type": str(robot_type),
                            "size": str(size),
                            "decode_strategy": decode,
                            "status": "ok" if status == 0 else f"failed:{status}",
                        }
                    )

    out_csv = Path(args.out_prefix).with_suffix(".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "robot_type", "size", "decode_strategy", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    return 0 if all(row["status"] in {"ok", "missing_model"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
