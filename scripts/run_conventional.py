from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from method_registry import REPO_ROOT


SOLVER_MODULES = {
    "gurobi": Path("GUROBI") / "GUROBI.py",
    "alns": Path("Algorithms") / "ALNS.py",
    "iga": Path("Algorithms") / "IG.py",
    "dabc": Path("Algorithms") / "DABC.py",
    "diwo": Path("Algorithms") / "DIWO.py",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run an exact or metaheuristic baseline through the packaged conventional implementation."
    )
    parser.add_argument("--solver", required=True, choices=sorted(SOLVER_MODULES))
    return parser.parse_known_args()


def main() -> int:
    args, extra = parse_args()
    conventional_dir = REPO_ROOT / "methods" / "conventional"
    script = conventional_dir / SOLVER_MODULES[args.solver]
    if not script.exists():
        raise FileNotFoundError(script)
    env = os.environ.copy()
    env["COHET_INSTANCE_DIR"] = str(REPO_ROOT / "instances" / "synthetic")
    command = [sys.executable, str(script), *extra]
    print("Running:", " ".join(command))
    return subprocess.call(command, cwd=conventional_dir, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
