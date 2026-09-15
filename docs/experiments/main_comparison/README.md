# Main comparison

The benchmark uses 100 fixed instances per cell at n = 10, 20, 50, 100 and k = 2, 3. The six DRL methods use Greedy and Sample-1280 decoding; conventional methods use the same instance sets.

| File | Contents |
|---|---|
| [normalized_candidates.csv](data/normalized_candidates.csv) | Collected feasible objectives, runtimes, repetitions, instance identifiers and status |
| [bks_per_instance.csv](data/bks_per_instance.csv) | Best-known feasible objective for each instance |
| [per_instance_method_rpd.csv](published/per_instance_method_rpd.csv) | Per-instance method objectives and RPD |
| [main_comparison_results.csv](published/main_comparison_results.csv) | Main comparison: RPD, standard deviation and computation time |
| [drl_timing_runs.csv](data/drl_timing_runs.csv) | Per-instance RTX 5090 timings for all six DRL methods |
| [meta_timing_runs.csv](data/meta_timing_runs.csv) | 104 measured timing runs for 26 metaheuristic cells |
| [timing_summary.csv](data/timing_summary.csv) | Computation-time summaries for all methods |
| [meta_incomplete_cells.csv](data/meta_incomplete_cells.csv) | Incomplete conventional-method cells |

BKS is the minimum feasible candidate objective for each instance. RPD is `100 * (objective - BKS) / BKS`, computed before averaging across instances. Metaheuristic repetitions are averaged within each instance for reporting; every feasible repetition remains eligible for BKS. Missing records are not evidence of infeasibility. Saved solver records remain under `source_evidence`.

Objective summaries use the available feasible instances: 93 for Gurobi at k=3, n=50 and 99 for ALNS at k=3, n=100. Failed, timed-out and missing outcomes remain documented in the source records and incomplete-cell summary. Algorithm settings are described in the [baseline guide](../../../methods/conventional/README.md).

The published DRL computation times were measured on an RTX 5090 and report the median of five complete run means, with 100 instances per run. The timing records contain 48,000 measurements across 96 method, mode and instance-scale settings. Times are reported in seconds per instance.

For 26 metaheuristic cells, time is the mean of four measured runs: instances I10 and I60, each repeated twice (104 rows in `meta_timing_runs.csv`). The other six metaheuristic cells—ALNS and IGA at k=2, n=100 and k=3, n=50 or 100—show the 3600 s budget, not an observed mean. Gurobi at n=100 likewise shows the 3600 s contract limit; this repository contains no source results for those cells, so the missing objectives cannot establish infeasibility. The `time_source`, `time_n` and `objective_n` fields in `timing_summary.csv` distinguish these cases.

For the published DRL measurements, each repetition ran the method's `eval.py` in a fresh process with evaluation batch size 1 and the evaluator's native sampling initialization. The timer covered decoding and result transfer, excluding model and dataset loading. No separate warmup pass or duration-based trimming was applied.

Rebuild the main table from the repository root, using a new output directory:

```powershell
conda run -n my310env python scripts/analysis/rebuild.py --output-dir outputs/main_comparison
```

This reads saved results and writes per-instance RPD, the method table and LaTeX. It does not run models or optimization.
