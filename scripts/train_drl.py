from __future__ import annotations

import argparse
import os
import subprocess
import sys

from method_registry import get_instance_root, get_method_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train Co-HeT or a DRL baseline. Extra arguments are forwarded to the method-specific run.py."
    )
    parser.add_argument("--method", required=True, choices=["cohet", "am", "hdrl", "tdrl", "mvmoe", "echo"])
    parser.add_argument("--robot_type", type=int, choices=[2, 3], default=2)
    return parser.parse_known_args()


def main() -> int:
    args, extra = parse_args()
    method_dir = get_method_dir(args.method, args.robot_type)
    env = os.environ.copy()
    env["COHET_INSTANCE_DIR"] = str(get_instance_root(args.robot_type))
    command = [sys.executable, "run.py", *extra]
    print("Running:", " ".join(command))
    print("Method directory:", method_dir)
    return subprocess.call(command, cwd=method_dir, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
