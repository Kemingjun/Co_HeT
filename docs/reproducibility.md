# Reproducibility notes

Run commands from the repository root. The training and evaluation interfaces are `scripts/train_drl.py`, `scripts/eval_drl.py` and `scripts/run_conventional.py`. Use each model's adjacent `args.json` for its batch size, precision and other training settings. The README presents logged training cost and validation mean cost in separate convergence figures.

The Full/Flat results report the observed quality and resource use of the released trained models. Model-specific training settings are provided alongside the [decoder_scaling checkpoints](../checkpoints/decoder_scaling).

The main synthetic benchmark has 100 fixed instances per n/k cell. Industrial simulation evaluation uses `real_world_test100_seed20260906`, with 100 generated test instances at each of n = 10, 20, 30, 40. Identify instances by dataset directory, size and index, since filenames repeat across datasets. The `real_world` paths and command-line option refer to industrial simulation assets. The [checkpoint guide](../checkpoints/README.md) identifies main and specialized models.

All routes start at the depot and end after the last assigned task. Travel time is Manhattan distance divided by robot speed. The main synthetic objective is `0.5 * total_travel_time + 0.5 * total_tardiness`, with reference speed 1. The [industrial objective](experiments/industrial/README.md) uses travel time at the carrier and forklift speeds. Coalition synchronization does not model collision avoidance, shared-space contention or failures during execution.

The main training seed is 1234; additional n=20, k=2 models use seeds 2345 and 3456. Inference sampling seeds are separate from training seeds. The fixed-instance main evaluator uses inference seed 20260827:

```powershell
conda run -n my310env python scripts/experiments/main/task_runner.py --phase quality --method Co-HeT --kappa 2 --size 20 --mode sample1280 --run-id cohet_n20_quality --gpu-id 0 --attempt-dir outputs/cohet_n20_quality
```

This command starts quality evaluation. Industrial training/evaluation commands and saved-result reconstruction are described on the [industrial page](experiments/industrial/README.md). The [main comparison page](experiments/main_comparison/README.md) describes BKS, RPD, the published timing protocol and table reconstruction.

Data-generation entry points are `scripts/generate_synthetic_extension.py` and `scripts/generate_industrial_testset.py`. Use `--help` for their generation options. New training and evaluation outputs should use a fresh run name and output directory.

## Multiseed statistics

The [multiseed materials](experiments/multiseed) contain `per_instance.csv` (3,600 per-instance statistical inputs), `summary.csv`, `pairwise_tests.csv` and [analysis settings](experiments/multiseed/analysis_config.json). Six methods are evaluated with training seeds 1234, 2345 and 3456 on 100 fixed n=20, k=2 instances under Greedy and Sample-1280. The CSV retains training seeds, evaluation seeds, instance indices and filenames separately; the settings link to the existing instance directory.

`summary.csv` reports the mean of the three training-seed means, their sample SD (`ddof=1`), and a 95% percentile CI from 10,000 crossed-bootstrap draws. Each draw independently resamples three seed indices and 100 instance indices with replacement. The configuration provides the actual per-cell bootstrap seeds, separate from training and inference seeds. Two-sided Wilcoxon signed-rank tests compare 100 per-instance objectives averaged over the three training seeds. Holm correction is applied to the five Co-HeT-versus-baseline comparisons separately within each inference mode. Rank-biserial effect sizes use baseline minus Co-HeT, so positive values favor Co-HeT.

Run the offline analysis from the repository root with a fresh output directory:

```powershell
conda run -n my310env python scripts/analysis/rebuild_statistics.py --topic multiseed --output-dir ../outputs/multiseed_statistics
```

This command reads the included CSV and writes `summary.csv` and `pairwise_tests.csv`; it does not train or evaluate models. Numerical reconstruction was checked with NumPy 1.26.4, pandas 2.1.1 and SciPy 1.15.3. Wilcoxon uses `zero_method="wilcox"`, `method="auto"` and average ranks for ties; all-zero paired differences give p=1 and effect=0.

## Generation-mechanism OOD

The [OOD materials](experiments/generation_ood) contain 33,600 per-instance statistical inputs for six methods, training seed 1234, n in {20, 50}, k in {2, 3}, and Greedy/Sample-1280. Each setting has 100 paired instances under ID and six generation shifts: D-IID, D-Bimodal, D-AntiDistance, P-TruncNormal, P-Lognormal and Joint. These inputs cover 48 OOD settings plus their eight ID settings; the existing [instance directories](../instances/generation_ood) are reused. The [analysis configuration](experiments/generation_ood/analysis_config.json) records evaluation settings and pairing rules.

Statistics use `objective`, computed as `0.5 * total_travel_time + 0.5 * total_tardiness` from the constructed schedule; `model_objective` retains the value returned by the model. At the reference speed of 1, the CSV's `distance` field equals total travel time, and `tardiness` records total tardiness. Pair instances by method, size, k, decoding mode and instance index, within this ID/OOD dataset. Compute each degradation as `100 * (F_OOD - F_ID) / F_ID` before taking means. The paper's mechanism-level values average the four size/type cells; overall values also average the six OOD mechanisms.

`summary.csv` gives objective means, sample SDs across instances, mean travel time (`distance_mean` at reference speed 1) and tardiness, paired degradation and within-setting ranks for all 336 method/setting combinations, including ID. `pairwise_tests.csv` contains the 240 OOD comparisons against Co-HeT. Two-sided Wilcoxon tests use the same zero/tie rules as above, with Holm correction over the five baseline comparisons **within each size/type/mechanism/decoding setting**. ID rows supply the paired reference for degradation and are not counted among these 240 tests.

```powershell
conda run -n my310env python scripts/analysis/rebuild_statistics.py --topic generation_ood --output-dir ../outputs/generation_ood_statistics
```

This offline command also prints the paper's paired-degradation aggregates.

The [OOD generator](../scripts/generate_generation_ood.py) and its [configuration](experiments/generation_ood/generator_config.json) implement the six generation shifts from the existing ID instances. Coordinates, task identities and required robot types remain paired; deadline-only shifts preserve processing times, processing-only shifts preserve deadlines, and Joint changes both. The instance master seed is 20260825; SHA-256 derives deterministic instance seeds.

```powershell
conda run -n my310env python scripts/generate_generation_ood.py --dry-run --output-dir ../outputs/generation_ood_instances
```

The dry run checks 400 existing ID workbooks without writing files. Omit `--dry-run` to generate 2,400 OOD workbooks in a new directory outside the repository. Existing output directories are refused.
