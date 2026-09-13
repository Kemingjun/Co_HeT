from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("objective", "travel_distance", "travel_time", "tardiness", "runtime_s")


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _bootstrap_ci(values: list[float], repetitions: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(repetitions)
    )
    low = means[int(0.025 * (repetitions - 1))]
    high = means[int(0.975 * (repetitions - 1))]
    return low, high


def _valid(record: dict) -> bool:
    return record.get("status") == "ok" and all(
        isinstance(record.get(metric), (int, float)) and math.isfinite(float(record[metric]))
        for metric in METRICS
    )


def build_summaries(
    records: list[dict],
    expected_tasks: list[dict],
    *,
    bootstrap_repetitions: int = 10000,
    bootstrap_seed: int = 20260825,
) -> dict[str, list[dict]]:
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    if len({row.get("dataset_id") for row in records}) > 1:
        raise ValueError("Mixed dataset identities cannot be summarized together")
    keys = [(row["size"], row["instance_index"], row["repetition"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate instance/repetition identity")
    dataset_id = records[0].get("dataset_id") if records else None
    per_rep = sorted(records, key=lambda row: (row["size"], row["instance_index"], row["repetition"]))
    by_key = {(row["size"], row["instance_index"], row["repetition"]): row for row in records}
    failed_or_missing = []
    for task in expected_tasks:
        key = (task["size"], task["instance_index"], task["repetition"])
        record = by_key.get(key)
        if record is None:
            failed_or_missing.append({**task, "status": "missing"})
        elif not _valid(record):
            failed_or_missing.append(record)

    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in records:
        if _valid(record):
            grouped[(record["size"], record["instance_index"])].append(record)
    per_instance = []
    for (size, index), rows in sorted(grouped.items()):
        row = {
            "dataset_id": dataset_id,
            "method": "ALNS",
            "size": size,
            "instance_index": index,
            "instance": rows[0].get("instance"),
            "repetitions_success": len(rows),
        }
        for metric in METRICS:
            mean, sd = _mean_sd([float(item[metric]) for item in rows])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_within_instance_sd"] = sd
        per_instance.append(row)

    rng = random.Random(bootstrap_seed)
    size_summary = []
    by_size: dict[int, list[dict]] = defaultdict(list)
    for row in per_instance:
        by_size[row["size"]].append(row)
    for size, rows in sorted(by_size.items()):
        summary = {"dataset_id": dataset_id, "method": "ALNS", "size": size, "instance_count": len(rows)}
        for metric in METRICS:
            values = [float(row[f"{metric}_mean"]) for row in rows]
            mean, sd = _mean_sd(values)
            low, high = _bootstrap_ci(values, bootstrap_repetitions, rng)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sd"] = sd
            summary[f"{metric}_ci95_low"] = low
            summary[f"{metric}_ci95_high"] = high
        size_summary.append(summary)
    return {
        "per_rep": per_rep,
        "per_instance": per_instance,
        "size_summary": size_summary,
        "failed_or_missing": failed_or_missing,
    }


def _read_records(run_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("n*/I*.json"))
    ]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                          for k, v in row.items()} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild ALNS summaries from immutable raw JSON records.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    args = parser.parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    expected = [
        {"size": size, "instance_index": index, "repetition": repetition}
        for size in config["sizes"]
        for index in range(config["instance_start"], config["instance_end"] + 1)
        for repetition in range(1, config["repetitions"] + 1)
    ]
    records = _read_records(args.run_dir)
    for row in records:
        if row.get("dataset_id") != config["dataset_id"]:
            raise ValueError("Result dataset identity mismatch")
        if row['size'] not in config['sizes'] or not config['instance_start'] <= row['instance_index'] <= config['instance_end']:
            raise ValueError("Result instance outside selected configuration")
        if not 1 <= row['repetition'] <= config['repetitions']:
            raise ValueError("Result repetition outside selected configuration")
    result = build_summaries(
        records,
        expected,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    missing = [row for row in result['failed_or_missing'] if row['status'] == 'missing']
    outputs = {
        "per_rep_results.csv": result["per_rep"] + missing,
        "size_summary.csv": result["size_summary"],
    }
    for name, rows in outputs.items():
        _write_csv(args.output_dir / name, rows)
    print(f"ALNS summaries written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
