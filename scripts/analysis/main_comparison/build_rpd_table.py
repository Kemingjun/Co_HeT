from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


INSTANCE_KEYS = ["kappa", "size", "instance_index"]
METHOD_INSTANCE_KEYS = [
    "candidate_source",
    "method",
    "mode",
    "kappa",
    "size",
    "instance_index",
    "instance_name",
]
CELL_KEYS = ["category", "method", "mode", "kappa", "size"]
SIZES = (10, 20, 50, 100)
KAPPAS = (2, 3)
METHOD_GROUPS = (
    ("MILP", ("Gurobi",)),
    ("Meta.", ("ALNS", "IGA", "DABC", "DIWO")),
    ("Greedy", ("AM", "MVMoE", "HDRL", "TDRL", "ECHO", "Co-HeT")),
    ("Sample-1280", ("AM", "MVMoE", "HDRL", "TDRL", "ECHO", "Co-HeT")),
)


def _normalise_mode(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["mode"] = result["mode"].fillna("").astype(str)
    return result


def _category_for_source(source: str, mode: str) -> str:
    if source == "Gurobi":
        return "MILP"
    if source == "Metaheuristic":
        return "Meta."
    if source == "DRL" and mode == "greedy":
        return "Greedy"
    if source == "DRL" and mode == "sample1280":
        return "Sample-1280"
    raise ValueError(f"Unsupported candidate source/mode: {source!r}/{mode!r}")


def compute_rpd_summary(
    candidates: pd.DataFrame, bks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = _normalise_mode(candidates)
    required_candidate_columns = set(METHOD_INSTANCE_KEYS + ["objective"])
    required_bks_columns = set(INSTANCE_KEYS + ["instance_name", "bks_objective"])
    if missing := required_candidate_columns.difference(candidates.columns):
        raise ValueError(f"Candidate columns missing: {sorted(missing)}")
    if missing := required_bks_columns.difference(bks.columns):
        raise ValueError(f"BKS columns missing: {sorted(missing)}")

    objectives = pd.to_numeric(candidates["objective"], errors="raise")
    if not np.isfinite(objectives).all() or (objectives <= 0).any():
        raise ValueError("All candidate objectives must be finite and positive")

    bks_values = pd.to_numeric(bks["bks_objective"], errors="raise")
    if not np.isfinite(bks_values).all() or (bks_values <= 0).any():
        raise ValueError("All BKS objectives must be finite and positive")
    if bks.duplicated(INSTANCE_KEYS).any():
        raise ValueError("BKS contains duplicate instance keys")

    per_instance = (
        candidates.assign(objective=objectives)
        .groupby(METHOD_INSTANCE_KEYS, as_index=False, dropna=False)
        .agg(mean_objective=("objective", "mean"), repetition_n=("objective", "size"))
    )
    merged = per_instance.merge(
        bks[INSTANCE_KEYS + ["instance_name", "bks_objective"]],
        on=INSTANCE_KEYS,
        how="left",
        validate="many_to_one",
        suffixes=("", "_bks"),
    )
    if merged["bks_objective"].isna().any():
        raise ValueError("At least one candidate instance has no BKS")
    if not (merged["instance_name"] == merged["instance_name_bks"]).all():
        raise ValueError("Candidate and BKS instance names do not match")

    merged["rpd"] = 100.0 * (
        merged["mean_objective"] - merged["bks_objective"]
    ) / merged["bks_objective"]
    merged["category"] = [
        _category_for_source(source, mode)
        for source, mode in zip(merged["candidate_source"], merged["mode"])
    ]
    per_instance_output = merged[
        CELL_KEYS
        + [
            "instance_index",
            "instance_name",
            "repetition_n",
            "mean_objective",
            "bks_objective",
            "rpd",
        ]
    ].sort_values(CELL_KEYS + ["instance_index"], ignore_index=True)

    summary = (
        per_instance_output.groupby(CELL_KEYS, as_index=False, dropna=False)
        .agg(
            rpd_n=("rpd", "size"),
            mean_rpd=("rpd", "mean"),
            sd_rpd=("rpd", "std"),
        )
        .sort_values(CELL_KEYS, ignore_index=True)
    )
    return per_instance_output, summary


def merge_rpd_with_times(rpd_summary: pd.DataFrame, time_summary: pd.DataFrame) -> pd.DataFrame:
    times = _normalise_mode(time_summary)
    result = times.merge(rpd_summary, on=CELL_KEYS, how="left", validate="one_to_one")
    return result[
        CELL_KEYS
        + [
            "rpd_n",
            "mean_rpd",
            "sd_rpd",
            "time_n",
            "mean_time_s",
            "time_source",
        ]
    ]


def annotate_rpd_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["rpd_rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    for _, group in ranked.groupby(["kappa", "size"], sort=False):
        tie_breakers = [
            column for column in ("category", "method", "mode") if column in group.columns
        ]
        valid = group[group["mean_rpd"].notna()].sort_values(
            ["mean_rpd", *tie_breakers], kind="stable"
        )
        for rank, index in enumerate(valid.index, start=1):
            ranked.loc[index, "rpd_rank"] = rank
    return ranked


def _mode_for_category(category: str) -> str:
    if category == "Greedy":
        return "greedy"
    if category == "Sample-1280":
        return "sample1280"
    return ""


def _format_rpd(value: float, sd: float, rank: object) -> str:
    if pd.isna(value):
        return "--"
    sd_text = "--" if pd.isna(sd) else f"{sd:.2f}"
    formatted = rf"{value:.2f} $\pm$ {sd_text}"
    if rank == 1:
        return rf"\textbf{{{formatted}}}"
    if rank == 2:
        return formatted + r"\(^{\dagger}\)"
    return formatted


def _format_time(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.2f}"


def render_latex(summary: pd.DataFrame) -> str:
    indexed = _normalise_mode(summary).set_index(CELL_KEYS, verify_integrity=True)
    lines = [
        r"\begin{table*}[ht]",
        r"\footnotesize",
        r"\centering",
        r"\caption{Comparison of Co-HeT With Baseline Methods on Instances With 2 and 3 Robot Types.}",
        r"\label{result_comparison}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\begin{tabular}{@{}llcc cc cc cc | cc cc cc cc@{}}",
        r"\Xhline{0.8pt}",
        r"\multirow{3}{*}{Category} & \multirow{3}{*}{Method}",
        r"& \multicolumn{8}{c|}{$\kappa=2$}",
        r"& \multicolumn{8}{c}{$\kappa=3$} \\",
        r"\cmidrule(r){3-10} \cmidrule(r){11-18}",
        r"& &",
        r"\multicolumn{2}{c}{$n=10$} &",
        r"\multicolumn{2}{c}{$n=20$} &",
        r"\multicolumn{2}{c}{$n=50$} &",
        r"\multicolumn{2}{c|}{$n=100$} &",
        r"\multicolumn{2}{c}{$n=10$} &",
        r"\multicolumn{2}{c}{$n=20$} &",
        r"\multicolumn{2}{c}{$n=50$} &",
        r"\multicolumn{2}{c}{$n=100$} \\",
        r"\cmidrule(r){3-4} \cmidrule(r){5-6} \cmidrule(r){7-8} \cmidrule(r){9-10}",
        r"\cmidrule(r){11-12} \cmidrule(r){13-14} \cmidrule(r){15-16} \cmidrule(r){17-18}",
        r"& & RPD & Time & RPD & Time & RPD & Time & RPD & Time",
        r"& RPD & Time & RPD & Time & RPD & Time & RPD & Time \\",
        r"\hline",
    ]

    for category, methods in METHOD_GROUPS:
        mode = _mode_for_category(category)
        for method_index, method in enumerate(methods):
            display_method = "Our" if method == "Co-HeT" else method
            if len(methods) == 1:
                row_start = f"{category} & {display_method}"
            elif method_index == 0:
                category_text = (
                    r"\makecell{Sample\\(1280)}" if category == "Sample-1280" else category
                )
                row_start = rf"\multirow{{{len(methods)}}}{{*}}{{{category_text}}} & {display_method}"
            else:
                row_start = f"& {display_method}"

            cells = []
            for kappa in KAPPAS:
                for size in SIZES:
                    key = (category, method, mode, kappa, size)
                    if key not in indexed.index:
                        rpd_value, rpd_sd, time_value, rank = math.nan, math.nan, math.nan, pd.NA
                    else:
                        record = indexed.loc[key]
                        rpd_value = record["mean_rpd"]
                        rpd_sd = record["sd_rpd"]
                        time_value = record["mean_time_s"]
                        rank = record["rpd_rank"]
                    cells.extend([_format_rpd(rpd_value, rpd_sd, rank), _format_time(time_value)])
            lines.append(row_start)
            lines.append("& " + " & ".join(cells[:8]))
            lines.append("& " + " & ".join(cells[8:]) + r" \\")
        lines.append(r"\hline" if category != "Sample-1280" else r"\Xhline{0.8pt}")

    lines.extend(
        [
            r"\end{tabular}",
            "",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize",
            r"Note: \textbf{Bold} indicates the best mean RPD; mark \(^{\dagger}\) denotes the second-best mean RPD. A dash indicates no reported value; consult the saved status and coverage records for the reason.",
            r"\end{minipage}",
            r"\vspace{-10pt}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"
