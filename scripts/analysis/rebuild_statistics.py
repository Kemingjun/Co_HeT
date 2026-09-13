"""Rebuild multiseed and generation-OOD statistics from per-instance CSV inputs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]


def paired_test(cohet, baseline, options=None):
    cohet, baseline = np.asarray(cohet, float), np.asarray(baseline, float)
    if cohet.ndim != 1 or cohet.shape != baseline.shape:
        raise ValueError("Expected equal-length paired vectors")
    differences = baseline - cohet
    if not np.isfinite(differences).all():
        raise ValueError("Non-finite paired objectives")
    nonzero = differences != 0
    if not nonzero.any():
        return {"p_raw": 1.0, "r_rb": 0.0}
    options = options or {"alternative": "two-sided", "zero_method": "wilcox", "method": "auto"}
    _, p_value = wilcoxon(differences, **options)
    ranks = rankdata(np.abs(differences[nonzero]), method="average")
    effect = (ranks[differences[nonzero] > 0].sum() - ranks[differences[nonzero] < 0].sum()) / ranks.sum()
    return {"p_raw": float(p_value), "r_rb": float(effect)}


def holm_adjust(p_values):
    values = np.asarray(p_values, float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Expected finite p-values in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    adjusted[order] = np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values)))).clip(max=1)
    return adjusted


def validate_input(data, config, topic):
    if topic == "multiseed":
        group_columns = ["method", "train_seed", "mode"]
        expected = set(itertools.product(config["methods"], config["training_seeds"], config["evaluation_modes"]))
        count = config["instance_count"]
        numeric = ["objective"]
    else:
        group_columns = ["method", "kappa", "size", "mechanism", "train_seed", "decode"]
        expected = set(itertools.product(config["methods"], config["kappas"], config["sizes"],
                                         config["mechanisms"], config["main_training_seeds"], config["decodes"]))
        count = config["instance_count_per_cell"]
        numeric = ["objective", "model_objective", "distance", "tardiness"]
    required = group_columns + ["instance_index", "instance_filename", "eval_seed"] + numeric
    if not set(required).issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(set(required) - set(data.columns))}")
    if data[required].isna().any().any() or not np.isfinite(data[numeric].to_numpy(float)).all():
        raise ValueError("Missing or non-finite input values; rows must not be silently dropped")
    actual = set(map(tuple, data[group_columns].drop_duplicates().to_numpy()))
    if actual != expected or len(data) != len(expected) * count:
        raise ValueError("Missing or unexpected method/seed/setting combinations or row count")
    if data.duplicated(group_columns + ["instance_index"]).any():
        raise ValueError("Duplicate instance within a setting")
    if not data.eval_seed.eq(config["sample_eval_seed"]).all():
        raise ValueError("Evaluation seed differs from the released configuration")
    for _, group in data.groupby(group_columns, sort=False):
        if sorted(group.instance_index.tolist()) != list(range(1, count + 1)):
            raise ValueError("Each setting must contain instance indices 1 through 100")
    if topic == "multiseed":
        n, k, robots = config["graph_size"], config["robot_type_num"], config["robot_count"]
        expected_names = data.instance_index.map(lambda i: f"N{n}_K{k}_M{robots}_I{i}.xlsx")
        if "instance_id" not in data or not data.instance_id.eq(data.instance_index.map(lambda i: f"I{i}")).all():
            raise ValueError("Instance IDs disagree with indices")
    else:
        expected_names = pd.Series([f"N{n}_K{k}_M{6*k}_I{i}.xlsx" for n, k, i in
                                   data[["size", "kappa", "instance_index"]].itertuples(index=False, name=None)], index=data.index)
        w = config["objective_weights"]
        if not np.allclose(data.objective, w["distance"] * data.distance + w["tardiness"] * data.tardiness,
                           rtol=0, atol=1e-6):
            raise ValueError("Objective is inconsistent with distance/tardiness and configured weights")
        if not (data.loc[data.mechanism == "ID", "objective"] > 0).all():
            raise ValueError("ID objectives must be positive for paired percentage degradation")
    if not data.instance_filename.eq(expected_names).all():
        raise ValueError("Instance filenames disagree with indices or scale/type configuration")


def crossed_summary(matrix, seed, replicates):
    seed_means = matrix.mean(axis=1)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for i in range(replicates):
        seed_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        instance_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        estimates[i] = matrix[np.ix_(seed_indices, instance_indices)].mean()
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {"mean": seed_means.mean(), "sd_seed": seed_means.std(ddof=1), "ci_low": low, "ci_high": high}


def compare_family(vectors, config, setting):
    cohet = vectors["Co-HeT"]
    rows = []
    for method in config["methods"]:
        if method == "Co-HeT":
            continue
        baseline = vectors[method]
        rows.append({**setting, "reference": "Co-HeT", "baseline": method, "n_pairs": len(cohet),
                     **paired_test(cohet, baseline, config["wilcoxon"]),
                     "mean_difference_baseline_minus_cohet": float((baseline - cohet).mean())})
    for row, p_value in zip(rows, holm_adjust([row["p_raw"] for row in rows])):
        row["p_holm"] = p_value
    return rows


def build_multiseed(data, config):
    validate_input(data, config, "multiseed")
    summaries, tests, vectors = [], [], {}
    for method in config["methods"]:
        for mode in config["evaluation_modes"]:
            selected = data[(data.method == method) & (data["mode"] == mode)]
            matrix = selected.pivot(index="train_seed", columns="instance_index", values="objective")
            matrix = matrix.loc[config["training_seeds"], list(range(1, config["instance_count"] + 1))].to_numpy(float)
            vectors[method, mode] = matrix.mean(axis=0)
            summaries.append({"method": method, "mode": mode, "n_training_seeds": len(matrix),
                              "n_instances": matrix.shape[1], **crossed_summary(
                                  matrix, config["bootstrap_cell_seeds"][method][mode], config["bootstrap_replicates"])})
    for mode in config["evaluation_modes"]:
        tests.extend(compare_family({m: vectors[m, mode] for m in config["methods"]}, config, {"mode": mode}))
    return pd.DataFrame(summaries), pd.DataFrame(tests)


def build_ood(data, config):
    validate_input(data, config, "generation_ood")
    data = data.sort_values(["method", "kappa", "size", "mechanism", "decode", "instance_index"])
    pair_keys = ["method", "kappa", "size", "decode", "instance_index"]
    id_data = data[data.mechanism == "ID"][pair_keys + ["objective"]].rename(columns={"objective": "id_objective"})
    paired = data.merge(id_data, on=pair_keys, how="left", validate="many_to_one")
    paired["degradation_percent"] = (paired.objective - paired.id_objective) / paired.id_objective * 100.0
    columns = ["method", "kappa", "size", "mechanism", "decode"]
    summaries = []
    for keys, group in paired.groupby(columns, sort=True):
        values = group.objective.to_numpy(float)
        summaries.append({**dict(zip(columns, keys)), "n_instances": len(group),
                          "objective_mean": values.mean(), "objective_sd": values.std(ddof=1),
                          "distance_mean": group.distance.mean(), "tardiness_mean": group.tardiness.mean(),
                          "id_degradation_percent_mean": group.degradation_percent.mean()})
    summary = pd.DataFrame(summaries)
    settings = ["kappa", "size", "mechanism", "decode"]
    summary["rank"] = summary.groupby(settings).objective_mean.rank(method="min").astype(int)
    tests = []
    for keys, group in data[data.mechanism != "ID"].groupby(settings, sort=True):
        vectors = {method: group[group.method == method].sort_values("instance_index").objective.to_numpy(float)
                   for method in config["methods"]}
        tests.extend(compare_family(vectors, config, dict(zip(settings, keys))))
    return summary, pd.DataFrame(tests)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", choices=["multiseed", "generation_ood"], required=True)
    parser.add_argument("--input-dir", type=Path, help="Directory containing per_instance.csv and analysis_config.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing directories are never overwritten")
    args = parser.parse_args(argv)
    source = args.input_dir or REPO_ROOT / "docs/experiments" / args.topic
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output_dir}")
    config = json.loads((source / "analysis_config.json").read_text(encoding="utf-8"))
    data = pd.read_csv(source / "per_instance.csv")
    build = build_multiseed if args.topic == "multiseed" else build_ood
    summary, tests = build(data, config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    tests.to_csv(args.output_dir / "pairwise_tests.csv", index=False)
    print(f"Read {len(data)} paired input rows; wrote {len(summary)} summary rows and {len(tests)} tests.")
    if args.topic == "generation_ood":
        ood = summary[summary.mechanism != "ID"]
        print("Co-HeT mean paired degradation (%) across size/type cells:")
        print(ood[ood.method == "Co-HeT"].groupby(["decode", "mechanism"]).id_degradation_percent_mean.mean().round(2).to_string())
        print("Mean paired degradation (%) across all six OOD mechanisms:")
        print(ood.groupby(["decode", "method"]).id_degradation_percent_mean.mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
