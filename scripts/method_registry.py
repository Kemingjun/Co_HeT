from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    display_name: str
    synthetic_dirs: dict[int, Path]
    real_world_dir: Path | None = None


METHODS: dict[str, MethodSpec] = {
    "cohet": MethodSpec(
        name="cohet",
        display_name="Co-HeT",
        synthetic_dirs={
            2: REPO_ROOT / "methods" / "learning" / "cohet" / "type_2",
            3: REPO_ROOT / "methods" / "learning" / "cohet" / "type_3",
        },
        real_world_dir=REPO_ROOT / "methods" / "real_world" / "cohet",
    ),
    "am": MethodSpec(
        name="am",
        display_name="AM",
        synthetic_dirs={
            2: REPO_ROOT / "methods" / "learning" / "am" / "type_2",
            3: REPO_ROOT / "methods" / "learning" / "am" / "type_3",
        },
        real_world_dir=REPO_ROOT / "methods" / "real_world" / "am",
    ),
    "hdrl": MethodSpec(
        name="hdrl",
        display_name="HDRL",
        synthetic_dirs={
            2: REPO_ROOT / "methods" / "learning" / "hdrl" / "type_2",
            3: REPO_ROOT / "methods" / "learning" / "hdrl" / "type_3",
        },
        real_world_dir=REPO_ROOT / "methods" / "real_world" / "hdrl",
    ),
    "tdrl": MethodSpec(
        name="tdrl",
        display_name="TDRL",
        synthetic_dirs={
            2: REPO_ROOT / "methods" / "learning" / "tdrl" / "type_2",
            3: REPO_ROOT / "methods" / "learning" / "tdrl" / "type_3",
        },
        real_world_dir=REPO_ROOT / "methods" / "real_world" / "tdrl",
    ),
    "mvmoe": MethodSpec(
        name="mvmoe",
        display_name="MVMoE",
        synthetic_dirs={
            2: REPO_ROOT / "methods" / "learning" / "mvmoe" / "type_2",
            3: REPO_ROOT / "methods" / "learning" / "mvmoe" / "type_3",
        },
        real_world_dir=REPO_ROOT / "methods" / "real_world" / "mvmoe",
    ),
    "echo": MethodSpec(
        name="echo",
        display_name="ECHO",
        synthetic_dirs={
            2: REPO_ROOT / "methods" / "learning" / "echo" / "type_2",
            3: REPO_ROOT / "methods" / "learning" / "echo" / "type_3",
        },
        real_world_dir=REPO_ROOT / "methods" / "real_world" / "echo",
    ),
}


def get_method(method: str) -> MethodSpec:
    key = method.lower()
    if key not in METHODS:
        supported = ", ".join(sorted(METHODS))
        raise ValueError(f"Unsupported method '{method}'. Supported methods: {supported}")
    return METHODS[key]


def get_method_dir(method: str, robot_type: int, real_world: bool = False) -> Path:
    spec = get_method(method)
    if real_world:
        if spec.real_world_dir is None:
            raise ValueError(f"{spec.display_name} does not provide a real-world implementation.")
        return spec.real_world_dir
    if robot_type not in spec.synthetic_dirs:
        raise ValueError(f"{spec.display_name} does not provide type-{robot_type} implementation.")
    return spec.synthetic_dirs[robot_type]


def get_instance_root(robot_type: int, real_world: bool = False) -> Path:
    if real_world:
        return REPO_ROOT / "instances" / "real_world" / "Instance_real_world"
    return REPO_ROOT / "instances" / "synthetic" / f"type_{robot_type}" / "Instance"


def infer_size_from_model(model_path: str | Path) -> int:
    path = Path(model_path)
    for part in reversed(path.parts):
        if part.startswith("size_"):
            return int(part.split("_", 1)[1])
    raise ValueError(f"Could not infer size from model path: {model_path}")


def default_dataset_name(robot_type: int, size: int, real_world: bool = False) -> str:
    if real_world:
        return f"RW_N{size}_K3_M16"
    if robot_type == 2:
        return f"N{size}_K2_M12"
    if robot_type == 3:
        return f"N{size}_K3_M18"
    raise ValueError("robot_type must be 2 or 3")
