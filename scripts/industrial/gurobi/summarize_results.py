from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
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
    return means[int(0.025 * (repetitions - 1))], means[int(0.975 * (repetitions - 1))]


def _feasible(record: dict) -> bool:
    return record.get("status") == "ok" and all(
        isinstance(record.get(metric), (int, float)) and math.isfinite(float(record[metric]))
        for metric in METRICS
    )


def _objective_consistent(record: dict) -> bool:
    explicit = record.get("objective_consistent")
    if isinstance(explicit, bool):
        return explicit
    error = record.get("objective_replay_error")
    return isinstance(error, (int, float)) and math.isfinite(float(error)) and float(error) <= 1e-6


def build_summaries(
    records: list[dict],
    expected_tasks: list[dict],
    *,
    bootstrap_repetitions: int = 10000,
    bootstrap_seed: int = 20260825,
) -> dict[str, list[dict]]:
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    keys = [(row['size'], row['instance_index']) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate instance identity')
    per_instance = sorted(records, key=lambda row: (row["size"], row["instance_index"]))
    by_key = {(row["size"], row["instance_index"]): row for row in records}
    failed_or_missing = []
    for task in expected_tasks:
        record = by_key.get((task["size"], task["instance_index"]))
        if record is None:
            failed_or_missing.append({**task, "status": "missing"})
        elif not _feasible(record):
            failed_or_missing.append(record)

    rng = random.Random(bootstrap_seed)
    size_summary = []
    for size in sorted({task["size"] for task in expected_tasks}):
        size_records = [row for row in records if row["size"] == size]
        feasible = [row for row in size_records if _feasible(row)]
        summary = {
            "method": "Gurobi",
            "size": size,
            "total_count": len(size_records),
            "feasible_count": len(feasible),
            "no_incumbent_count": sum(row.get("status") == "no_incumbent" for row in size_records),
            "failed_count": sum(row.get("status") == "failed" for row in size_records),
            "objective_inconsistent_count": sum(not _objective_consistent(row) for row in feasible),
        }
        replay_errors = [
            float(row["objective_replay_error"])
            for row in feasible
            if isinstance(row.get("objective_replay_error"), (int, float))
            and math.isfinite(float(row["objective_replay_error"]))
        ]
        summary["objective_replay_error_mean"] = (
            statistics.fmean(replay_errors) if replay_errors else math.nan
        )
        summary["objective_replay_error_max"] = max(replay_errors) if replay_errors else math.nan
        for metric in METRICS:
            values = [float(row[metric]) for row in feasible]
            if values:
                mean, sd = _mean_sd(values)
                low, high = _bootstrap_ci(values, bootstrap_repetitions, rng)
            else:
                mean = sd = low = high = math.nan
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sd"] = sd
            summary[f"{metric}_ci95_low"] = low
            summary[f"{metric}_ci95_high"] = high
        size_summary.append(summary)

    counts = Counter((row["size"], row.get("solver_status_name", "UNKNOWN")) for row in records)
    status_summary = [
        {"size": size, "solver_status_name": status, "count": count}
        for (size, status), count in sorted(counts.items())
    ]
    objective_inconsistencies = [
        row for row in per_instance if _feasible(row) and not _objective_consistent(row)
    ]
    return {
        "per_instance": per_instance,
        "size_summary": size_summary,
        "status_summary": status_summary,
        "objective_inconsistencies": objective_inconsistencies,
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
    parser = argparse.ArgumentParser(description="Rebuild Gurobi summaries from immutable raw JSON records.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    args = parser.parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    expected = [
        {"size": size, "instance_index": index}
        for size in config["sizes"]
        for index in range(config["instance_start"], config["instance_end"] + 1)
    ]
    records = _read_records(args.run_dir)
    for row in records:
        if row.get('dataset_id') != config['dataset_id']:
            raise ValueError('Result dataset identity mismatch')
        if row['size'] not in config['sizes'] or not config['instance_start'] <= row['instance_index'] <= config['instance_end']:
            raise ValueError('Result instance outside selected configuration')
    result = build_summaries(
        records,
        expected,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    missing = [row for row in result['failed_or_missing'] if row['status'] == 'missing']
    outputs = {
        "per_instance_results.csv": result["per_instance"] + missing,
        "size_summary.csv": result["size_summary"],
    }
    for name, rows in outputs.items():
        _write_csv(args.output_dir / name, rows)
    print(f"Gurobi summaries written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
