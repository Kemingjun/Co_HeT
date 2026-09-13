"""Run the published synthetic Gurobi and metaheuristic baselines."""

import sys
from pathlib import Path

CONVENTIONAL_ROOT = Path(__file__).resolve().parents[1] / "methods" / "conventional"
sys.path.insert(0, str(CONVENTIONAL_ROOT))

from conventional_runner import build_parser, main


if __name__ == "__main__":
    raise SystemExit(main())
