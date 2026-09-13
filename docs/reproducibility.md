# Reproducibility notes

Run commands from the repository root. The training and evaluation interfaces are `scripts/train_drl.py`, `scripts/eval_drl.py` and `scripts/run_conventional.py`. Use each model's adjacent `args.json` for its batch size, precision and other training settings. Training budgets depend on these settings as well as epoch counts. The README presents logged training cost and validation mean cost in separate convergence figures.

The Full/Flat decoder results compare the quality and resource use of existing trained models. Training settings differ: Flat uses batch 512 versus Full's 1024 at n=30, AMP FP16 versus FP32 at n=40, and batch 800 with AMP FP16 versus Full's batch 1024 at n=50. These differences preclude a strictly controlled single-factor decoder comparison. The adjacent configurations in [decoder_scaling](../checkpoints/decoder_scaling) preserve the settings.

The main synthetic benchmark has 100 fixed instances per n/k cell. Industrial simulation evaluation uses `real_world_test100_seed20260906`, with 100 generated test instances at each of n = 10, 20, 30, 40. Identify instances by dataset directory, size and index, since filenames repeat across datasets. The `real_world` paths and command-line option refer to industrial simulation assets. The [checkpoint guide](../checkpoints/README.md) identifies main and specialized models.

All routes start at the depot and end after the last assigned task. Travel uses Manhattan distance. The main synthetic objective is `0.5 * distance + 0.5 * tardiness`, with reference speed 1. The [industrial objective](experiments/industrial/README.md) uses travel time at the carrier and forklift speeds. Coalition synchronization does not model collision avoidance, shared-space contention or failures during execution.

The main training seed is 1234; additional n=20, k=2 models use seeds 2345 and 3456. Inference sampling seeds are separate from training seeds. The fixed-instance main evaluator uses inference seed 20260827:

```powershell
conda run -n my310env python scripts/experiments/main/task_runner.py --phase quality --method Co-HeT --kappa 2 --size 20 --mode sample1280 --run-id cohet_n20_quality --gpu-id 0 --attempt-dir outputs/cohet_n20_quality
```

This command starts evaluation. Its `timing` phase invokes the method's original `eval.py` in five separate processes, with 100 instances per run and evaluation batch size 1. It checks GPU contention and reports the median of the five run means, without trimming durations. Timing uses the evaluator's native timer and sampling initialization; the fixed inference seed above applies to the quality phase. The published timings were measured on an RTX 5090. Industrial training/evaluation commands and saved-result reconstruction are described on the [industrial page](experiments/industrial/README.md). The [main comparison page](experiments/main_comparison/README.md) describes BKS, RPD and table reconstruction.

Data-generation entry points are `scripts/generate_synthetic_extension.py` and `scripts/generate_industrial_testset.py`. Use `--help` for their generation options. New training and evaluation outputs should use a fresh run name and output directory.
