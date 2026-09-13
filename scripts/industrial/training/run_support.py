from __future__ import annotations

import json
import os
import platform
import socket
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def load_experiment_config() -> dict:
    return json.loads((PACKAGE_ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


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


def environment_snapshot() -> dict:
    snapshot = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "pid": os.getpid(),
    }
    try:
        import torch

        snapshot.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        snapshot["torch"] = None
    return snapshot
