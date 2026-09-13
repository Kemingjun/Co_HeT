"""Generate synthetic benchmark instances I21-I100."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


EXPECTED_COLUMNS = [
    "task_index",
    "source_x",
    "source_y",
    "deadline",
    "t_operation",
    "required_robot",
]
SIZES = (10, 20, 50, 100)
KAPPAS = (2, 3)
LEGACY_START = 1
LEGACY_END = 20
ROUND2_END = 100
DEFAULT_MASTER_SEED = 20260825
SEED_NAMESPACE = "cohet-r2"
PROCESSING_RANGES = {
    2: ((0.3, 0.7), (0.8, 1.2)),
    3: ((0.4, 0.8), (0.6, 1.0), (0.8, 1.2)),
}


class GenerationError(RuntimeError):
    """Raised when an instance or frozen artifact violates the generation contract."""


def derive_seed(master_seed: int, *, kappa: int, n: int, index: int) -> int:
    material = (
        f"{SEED_NAMESPACE}|{master_seed}|kappa={kappa}|n={n}|i={index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def generate_instance(*, n: int, kappa: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if kappa not in PROCESSING_RANGES:
        raise ValueError(f"Unsupported kappa: {kappa}")

    rng = random.Random(seed)
    deadlines = [0.5 * task + rng.uniform(-0.2, 0.2) for task in range(1, n + 1)]
    rng.shuffle(deadlines)

    rows: list[dict[str, Any]] = []
    for task in range(1, n + 1):
        rows.append(
            {
                "task_index": task,
                "source_x": rng.random(),
                "source_y": rng.random(),
                "deadline": deadlines[task - 1],
                "t_operation": [
                    rng.uniform(low, high) for low, high in PROCESSING_RANGES[kappa]
                ],
                "required_robot": [1] * kappa,
            }
        )
    return rows


def _parse_list(value: Any, *, field: str, path: Path) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as exc:
        raise GenerationError(f"Invalid {field} in {path}: {value!r}") from exc
    if not isinstance(parsed, list):
        raise GenerationError(f"{field} is not a list in {path}: {value!r}")
    return parsed


def read_instance(path: Path) -> list[dict[str, Any]]:
    try:
        frame = pd.read_excel(path)
    except Exception as exc:
        raise GenerationError(f"Cannot read workbook {path}: {exc}") from exc
    if frame.columns.tolist() != EXPECTED_COLUMNS:
        raise GenerationError(
            f"Unexpected columns in {path}: {frame.columns.tolist()}"
        )

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        rows.append(
            {
                "task_index": int(record["task_index"]),
                "source_x": float(record["source_x"]),
                "source_y": float(record["source_y"]),
                "deadline": float(record["deadline"]),
                "t_operation": [
                    float(value)
                    for value in _parse_list(
                        record["t_operation"], field="t_operation", path=path
                    )
                ],
                "required_robot": [
                    int(value)
                    for value in _parse_list(
                        record["required_robot"], field="required_robot", path=path
                    )
                ],
            }
        )
    return rows


def validate_rows(rows: Sequence[dict[str, Any]], *, n: int, kappa: int) -> None:
    if kappa not in PROCESSING_RANGES:
        raise GenerationError(f"Unsupported kappa: {kappa}")
    if len(rows) != n:
        raise GenerationError(f"Expected {n} rows, found {len(rows)}")

    for position, row in enumerate(rows, start=1):
        if list(row) != EXPECTED_COLUMNS:
            raise GenerationError(f"Unexpected field order in row {position}: {list(row)}")
        if row["task_index"] != position:
            raise GenerationError(
                f"Expected task_index {position}, found {row['task_index']}"
            )
        if not (0.0 <= row["source_x"] < 1.0 and 0.0 <= row["source_y"] < 1.0):
            raise GenerationError(f"Coordinate out of [0,1) in row {position}")
        if row["required_robot"] != [1] * kappa:
            raise GenerationError(f"Invalid required_robot in row {position}")
        if len(row["t_operation"]) != kappa:
            raise GenerationError(f"Invalid t_operation length in row {position}")
        for value, (low, high) in zip(
            row["t_operation"], PROCESSING_RANGES[kappa]
        ):
            if not low <= value <= high:
                raise GenerationError(
                    f"Processing time {value} outside [{low},{high}] in row {position}"
                )

    for position, deadline in enumerate(
        sorted(float(row["deadline"]) for row in rows), start=1
    ):
        low = 0.5 * position - 0.2
        high = 0.5 * position + 0.2
        if not low <= deadline <= high:
            raise GenerationError(
                f"Deadline {deadline} outside [{low},{high}] at sorted position {position}"
            )


def _robot_count(kappa: int) -> int:
    return 12 if kappa == 2 else 18


def instance_directory(instances_root: Path, *, n: int, kappa: int) -> Path:
    return (
        instances_root
        / f"type_{kappa}"
        / "Instance"
        / f"N{n}_K{kappa}_M{_robot_count(kappa)}"
    )


def instance_path(instances_root: Path, *, n: int, kappa: int, index: int) -> Path:
    stem = f"N{n}_K{kappa}_M{_robot_count(kappa)}_I{index}"
    return instance_directory(instances_root, n=n, kappa=kappa) / f"{stem}.xlsx"


def write_instance(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_excel(path, index=False)


def generation_config(master_seed: int) -> dict[str, Any]:
    return {
        "dataset_id": "synthetic",
        "sizes": list(SIZES),
        "kappas": list(KAPPAS),
        "count_per_cell": ROUND2_END,
        "legacy": {"index_range": [LEGACY_START, LEGACY_END], "master_seed": None},
        "extension": {
            "index_range": [LEGACY_END + 1, ROUND2_END],
            "master_seed": master_seed,
            "seed_derivation": (
                'SHA256("cohet-r2|{master_seed}|kappa={kappa}|n={n}|i={index}") '
                "first 8 bytes, big-endian"
            ),
        },
        "distribution": {
            "coordinates": "independent Uniform[0,1), no minimum spacing",
            "deadlines": "0.5,1.0,...,0.5n + independent Uniform[-0.2,0.2], then shuffled",
            "processing_time": {
                f"kappa_{kappa}": [list(bounds) for bounds in ranges]
                for kappa, ranges in PROCESSING_RANGES.items()
            },
            "required_robot": {f"kappa_{kappa}": [1] * kappa for kappa in KAPPAS},
            "distance": "not computed by the instance generator",
        },
        "columns": EXPECTED_COLUMNS,
    }


def _execute(
    *,
    instances_root: Path,
    start_index: int,
    end_index: int,
    master_seed: int,
    mode: str,
) -> dict[str, Any]:
    if mode not in ("dry-run", "write"):
        raise GenerationError(f"Unsupported mode: {mode}")
    if not LEGACY_END < start_index <= end_index <= ROUND2_END:
        raise GenerationError(
            f"Index range must satisfy {LEGACY_END} < start <= end <= {ROUND2_END}"
        )
    config_path = instances_root / "generation_config.json"
    config = generation_config(master_seed)
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise GenerationError(f"Generation parameters conflict: {config_path}")

    # Check the preserved prefix and any existing extensions without replacing them.
    for kappa in KAPPAS:
        for n in SIZES:
            directory = instance_directory(instances_root, n=n, kappa=kappa)
            if not directory.is_dir():
                raise GenerationError(f"Missing instance directory: {directory}")
            for index in range(LEGACY_START, LEGACY_END + 1):
                validate_rows(
                    read_instance(instance_path(instances_root, n=n, kappa=kappa, index=index)),
                    n=n, kappa=kappa,
                )
            prefix = f"N{n}_K{kappa}_M{_robot_count(kappa)}_I"
            for path in directory.glob("*.xlsx"):
                suffix = path.stem[len(prefix):]
                if (not path.stem.startswith(prefix) or not suffix.isdigit()
                        or not LEGACY_START <= int(suffix) <= ROUND2_END):
                    raise GenerationError(f"Unexpected workbook name: {path}")
                if int(suffix) > LEGACY_END:
                    validate_rows(read_instance(path), n=n, kappa=kappa)

    missing = []
    skipped = 0
    for kappa in KAPPAS:
        for n in SIZES:
            for index in range(start_index, end_index + 1):
                target = instance_path(instances_root, n=n, kappa=kappa, index=index)
                if target.exists():
                    skipped += 1
                else:
                    seed = derive_seed(master_seed, kappa=kappa, n=n, index=index)
                    rows = generate_instance(n=n, kappa=kappa, seed=seed)
                    validate_rows(rows, n=n, kappa=kappa)
                    missing.append((target, rows, n, kappa))

    if mode == "dry-run":
        print(f"DRY RUN OK: legacy=160, would_generate={len(missing)}, existing={skipped}")
        return {"generated": 0, "skipped": skipped, "missing": len(missing)}

    generated = 0
    with tempfile.TemporaryDirectory(prefix="cohet_r2_instances_") as temp_name:
        temp_root = Path(temp_name)
        staged = []
        for target, rows, n, kappa in missing:
            staged_path = temp_root / target.name
            write_instance(staged_path, rows)
            validate_rows(read_instance(staged_path), n=n, kappa=kappa)
            staged.append((staged_path, target))
        for staged_path, target in staged:
            if target.exists():
                raise GenerationError(f"Target appeared during generation: {target}")
            # Exclusive creation preserves existing workbooks even if another process writes.
            with target.open("xb") as stream, staged_path.open("rb") as source:
                shutil.copyfileobj(source, stream)
            generated += 1

    if not config_path.exists():
        with config_path.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
    print(f"WRITE OK: generated={generated}, existing={skipped}")
    return {"generated": generated, "skipped": skipped, "missing": 0}


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Append deterministic I21-I100 synthetic HRSP instances."
    )
    parser.add_argument("--start-index", type=int, default=21)
    parser.add_argument("--end-index", type=int, default=100)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument(
        "--instances-root",
        type=Path,
        default=repo_root / "instances" / "synthetic",
        help=argparse.SUPPRESS,
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = "write" if args.write else "dry-run"
    try:
        _execute(
            instances_root=args.instances_root.resolve(),
            start_index=args.start_index,
            end_index=args.end_index,
            master_seed=args.master_seed,
            mode=mode,
        )
    except GenerationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
