from __future__ import annotations

import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
INSTANCE_DIR = REPO_ROOT / "instances/real_world_test100_seed20260906"


def load_experiment_config() -> dict:
    return json.loads((PACKAGE_ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


CONFIG = load_experiment_config()
ALLOWED_SIZES = tuple(CONFIG['sizes'])
INSTANCE_START = CONFIG['instance_start']
INSTANCE_END = CONFIG['instance_end']
INSTANCE_COUNT = CONFIG['instances_per_size']
if INSTANCE_END - INSTANCE_START + 1 != INSTANCE_COUNT:
    raise ValueError('Inconsistent frozen instance range')


def write_json_once(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_or_validate_config(path: Path, payload: dict) -> None:
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Configuration conflict for existing run: {path}")
        return
    write_json_once(path, payload)


# The dataset selector is shared by the industrial entry points.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from industrial_data import dataset_identity, select_instances, validate_instance
