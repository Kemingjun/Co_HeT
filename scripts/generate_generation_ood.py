"""Regenerate six paired OOD mechanisms from the published ID workbooks."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "docs/experiments/generation_ood/generator_config.json"
MECHANISMS = (
    "D-IID", "D-Bimodal", "D-AntiDistance", "P-TruncNormal", "P-Lognormal", "Joint",
)
PROCESS_RANGES = {
    2: ((0.3, 0.7), (0.8, 1.2)),
    3: ((0.4, 0.8), (0.6, 1.0), (0.8, 1.2)),
}
# These are the published parameters, checked against the JSON before generation.
# The sampling functions below retain their original arithmetic and RNG call order.
MECHANISM_PARAMETERS = {
    "D-IID": {
        "distribution": "uniform", "low": 0.3,
        "high_n_coefficient": 0.5, "high_offset": 0.2,
    },
    "D-Bimodal": {
        "low_component_probability": 0.5,
        "low_beta_shapes": [2, 8], "high_beta_shapes": [8, 2],
        "offset": 0.3, "scale_n_coefficient": 0.5, "scale_offset": -0.1,
        "sampling_order": ["component_choices", "low_component", "high_component"],
    },
    "D-AntiDistance": {
        "distance": "source_x + source_y",
        "assignment": "ascending_ID_deadlines_to_descending_distance",
        "distance_sort": "stable", "shuffled_fraction": 0.3,
        "subset_size": "max(2, ceil(shuffled_fraction * n))",
        "subset_with_replacement": False,
        "perturbation": "shuffle_deadlines_within_selected_subset",
    },
    "P-TruncNormal": {
        "mean": "processing_range_midpoint", "standard_deviation": 0.1,
        "support": "processing_range_inclusive", "max_attempts_per_value": 10000,
        "sampling_order": "task_then_function",
    },
    "P-Lognormal": {
        "arithmetic_mean": "processing_range_midpoint", "coefficient_of_variation": 0.25,
        "sigma_log": "sqrt(log1p(CV * CV))",
        "mu_log": "log(arithmetic_mean) - 0.5 * sigma_log * sigma_log",
        "truncate": False, "sampling_order": "task_then_function",
    },
    "Joint": {
        "ordered_mechanisms": ["D-IID", "P-Lognormal"], "shared_rng_stream": True,
    },
}
EXPECTED_COLUMNS = (
    "task_index", "source_x", "source_y", "deadline", "t_operation", "required_robot",
)


def derive_seed(master_seed: int, mechanism: str, kappa: int, size: int, index: int) -> int:
    material = f"cohet-ood|{master_seed}|{mechanism}|kappa={kappa}|n={size}|i={index}"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _truncated_normal(rng: np.random.Generator, mean: float, sigma: float,
                      low: float, high: float) -> float:
    for _ in range(10000):
        value = float(rng.normal(mean, sigma))
        if low <= value <= high:
            return value
    raise RuntimeError("Unable to sample truncated normal")


def _processing_values(rng: np.random.Generator, n: int, kappa: int,
                       lognormal: bool) -> list[list[float]]:
    rows = []
    sigma_log = math.sqrt(math.log1p(0.25 * 0.25))
    for _ in range(n):
        values = []
        for low, high in PROCESS_RANGES[kappa]:
            mean = (low + high) / 2.0
            if lognormal:
                mu_log = math.log(mean) - 0.5 * sigma_log * sigma_log
                value = float(rng.lognormal(mu_log, sigma_log))
            else:
                value = _truncated_normal(rng, mean, 0.1, low, high)
            values.append(value)
        rows.append(values)
    return rows


def _iid_deadlines(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(0.3, 0.5 * n + 0.2, size=n)


def _bimodal_deadlines(rng: np.random.Generator, n: int) -> np.ndarray:
    choose_low = rng.random(n) < 0.5
    samples = np.empty(n, dtype=float)
    low_count = int(choose_low.sum())
    samples[choose_low] = rng.beta(2, 8, size=low_count)
    samples[~choose_low] = rng.beta(8, 2, size=n - low_count)
    return 0.3 + samples * (0.5 * n - 0.1)


def _anti_distance_deadlines(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    distance = frame["source_x"].to_numpy(float) + frame["source_y"].to_numpy(float)
    farthest_first = np.argsort(-distance, kind="stable")
    ordered_deadlines = np.sort(frame["deadline"].to_numpy(float))
    deadlines = np.empty(len(frame), dtype=float)
    deadlines[farthest_first] = ordered_deadlines
    selected_count = max(2, int(math.ceil(0.30 * len(frame))))
    selected = rng.choice(len(frame), size=selected_count, replace=False)
    shuffled = deadlines[selected].copy()
    rng.shuffle(shuffled)
    deadlines[selected] = shuffled
    return deadlines


def generate_variant(base: pd.DataFrame, mechanism: str, kappa: int, seed: int) -> pd.DataFrame:
    if mechanism not in MECHANISMS:
        raise ValueError(f"Unknown OOD mechanism: {mechanism}")
    if kappa not in PROCESS_RANGES:
        raise ValueError(f"Unsupported kappa: {kappa}")
    frame = base.copy(deep=True)
    frame["t_operation"] = [list(value) for value in base["t_operation"]]
    frame["required_robot"] = [list(value) for value in base["required_robot"]]
    rng = np.random.default_rng(seed)
    if mechanism in {"D-IID", "Joint"}:
        frame["deadline"] = _iid_deadlines(rng, len(frame))
    elif mechanism == "D-Bimodal":
        frame["deadline"] = _bimodal_deadlines(rng, len(frame))
    elif mechanism == "D-AntiDistance":
        frame["deadline"] = _anti_distance_deadlines(frame, rng)
    if mechanism == "P-TruncNormal":
        frame["t_operation"] = _processing_values(rng, len(frame), kappa, lognormal=False)
    elif mechanism in {"P-Lognormal", "Joint"}:
        frame["t_operation"] = _processing_values(rng, len(frame), kappa, lognormal=True)
    return frame[list(base.columns)]


def dataset_name(size: int, kappa: int) -> str:
    return f"N{size}_K{kappa}_M{12 if kappa == 2 else 18}"


def instance_index(path: Path) -> int:
    match = re.search(r"_I(\d+)\.xlsx$", path.name)
    if match is None:
        raise ValueError(f"Unexpected instance filename: {path.name}")
    return int(match.group(1))


def read_instance(path: Path, kappa: int) -> pd.DataFrame:
    frame = pd.read_excel(path)
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
    for column in ("t_operation", "required_robot"):
        frame[column] = frame[column].map(
            lambda value: ast.literal_eval(value) if isinstance(value, str) else list(value)
        )
    if len(frame) not in {20, 50}:
        raise ValueError(f"Unexpected task count in {path}: {len(frame)}")
    if frame.task_index.tolist() != list(range(1, len(frame) + 1)):
        raise ValueError(f"task_index is not 1..n in {path}")
    if any(len(values) != kappa for values in frame.t_operation):
        raise ValueError(f"Processing vector length mismatch in {path}")
    if any(values != [1] * kappa for values in frame.required_robot):
        raise ValueError(f"required_robot mismatch in {path}")
    return frame


def iter_id_instances(id_dir: Path, config: dict):
    for kappa in config["kappas"]:
        for size in config["sizes"]:
            folder = id_dir / f"kappa_{kappa}" / dataset_name(size, kappa)
            files = sorted(folder.glob("*.xlsx"), key=instance_index)
            if [instance_index(path) for path in files] != list(range(1, 101)):
                raise ValueError(f"ID suite is not exact I1-I100: {folder}")
            for path in files:
                yield kappa, size, instance_index(path), path


def write_instance(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy(deep=True)
    output["t_operation"] = output["t_operation"].map(lambda values: str(list(map(float, values))))
    output["required_robot"] = output["required_robot"].map(lambda values: str(list(map(int, values))))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        output.to_excel(stream, index=False, engine="openpyxl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Generator JSON; relative id_directory is resolved from the repository root")
    parser.add_argument("--id-dir", type=Path,
                        help="Existing ID directory containing kappa_2 and kappa_3 (overrides config)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory outside the repository and ID directory; must not exist")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and show count without writing")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    expected = {
        "master_seed": 20260825, "sizes": [20, 50], "kappas": [2, 3],
        "mechanisms": list(MECHANISMS), "instances_per_cell": 100,
        "processing_ranges": {str(k): [list(bounds) for bounds in ranges]
                              for k, ranges in PROCESS_RANGES.items()},
        "mechanism_parameters": MECHANISM_PARAMETERS,
    }
    if not isinstance(config, dict) or set(config) != set(expected) | {"id_directory"}:
        parser.error("Config must contain exactly the published keys, including id_directory")
    for key, value in expected.items():
        if config.get(key) != value:
            parser.error(f"The published generator requires {key}={value!r}")
    if not isinstance(config["id_directory"], str) or not config["id_directory"].strip():
        parser.error("id_directory must be a nonempty path string")
    id_dir = (args.id_dir if args.id_dir is not None else ROOT / config["id_directory"]).resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error("--output-dir must not already exist")
    if output_dir.is_relative_to(ROOT) or output_dir.is_relative_to(id_dir):
        parser.error("--output-dir must be outside the repository and ID directory")
    inputs = list(iter_id_instances(id_dir, config))
    # Validate every input before creating output directories.
    bases = []
    for kappa, size, index, path in inputs:
        base = read_instance(path, kappa)
        if len(base) != size:
            raise ValueError(f"Task count differs from the directory size: {path}")
        bases.append((kappa, size, index, path, base))
    count = len(bases) * len(MECHANISMS)
    if args.dry_run:
        print(f"Would generate {count} OOD workbooks from {len(bases)} existing ID workbooks into {output_dir}")
        return 0
    output_dir.mkdir(parents=True, exist_ok=False)
    for kappa, size, index, path, base in bases:
        for mechanism in MECHANISMS:
            seed = derive_seed(config["master_seed"], mechanism, kappa, size, index)
            frame = generate_variant(base, mechanism, kappa, seed)
            target = output_dir / mechanism / f"kappa_{kappa}" / dataset_name(size, kappa) / path.name
            write_instance(target, frame)
    print(f"Generated {count} OOD workbooks in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
