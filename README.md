# Co-HeT

Code and pretrained checkpoints for **Co-HeT: A Transformer-based Deep Reinforcement Learning Approach for Collaborative Heterogeneous Robot Scheduling**.

This repository is organized as a reproducible release package. It includes the proposed Co-HeT model, adapted DRL baselines, conventional baselines, synthetic and real-world instances, representative figures, a Webots simulation video, and final `epoch-99` checkpoints.

<p align="center">
  <img src="docs/figures/architecture.png" width="760">
</p>

## Highlights

- Collaborative heterogeneous robot scheduling with synchronized task-coalition decisions.
- Co-HeT model with task-robot mutual contextual encoding and composite-action decoding.
- Adapted DRL baselines: AM, HDRL, TDRL, MVMoE, and ECHO.
- Conventional baselines: Gurobi, ALNS, IGA, DABC, and DIWO.
- Synthetic, robustness, and industrial real-world case-study instances.
- Webots simulation demo: `media/cohet_real_world_factory.mp4`.

## Installation

```bash
conda env create -f environment.yml
conda activate cohet
```

If you use an existing environment, install the required packages manually:

```bash
pip install torch numpy pandas tqdm openpyxl tensorboard_logger
```

The checkpoint and video files are tracked with Git LFS:

```bash
git lfs install
git lfs pull
```

## Repository Structure

```text
methods/learning/      Co-HeT and adapted DRL baselines
methods/conventional/  Gurobi and metaheuristic baselines
checkpoints/           Final epoch-99 checkpoints
instances/             Synthetic, robustness, and real-world instances
scripts/               Unified training, evaluation, and benchmark entry points
docs/figures/          Public figures used in the README and result overview
media/                 Webots simulation video
results/               Output directory for reproduced results
```

## Evaluate Co-HeT

Greedy decoding on the 20-task, two-robot-type benchmark:

```bash
python scripts/eval_drl.py \
  --method cohet \
  --robot_type 2 \
  --dataset Synthetic_Dataset \
  --model checkpoints/cohet/type_2/size_20 \
  --decode_strategy greedy \
  --eval_batch_size 1
```

Sampling with 1280 candidate solutions:

```bash
python scripts/eval_drl.py \
  --method cohet \
  --robot_type 2 \
  --dataset Synthetic_Dataset \
  --model checkpoints/cohet/type_2/size_20 \
  --decode_strategy sample \
  --width 1280 \
  --eval_batch_size 1
```

To evaluate another scale, change the checkpoint folder:

```text
checkpoints/cohet/type_2/size_10
checkpoints/cohet/type_2/size_50
checkpoints/cohet/type_2/size_100
checkpoints/cohet/type_3/size_20
```

The dataset family is resolved automatically from `--robot_type` and the checkpoint size.

## Evaluate DRL Baselines

The repository includes CHRSP-adapted DRL baselines while preserving their core design principles:

- **AM**: attention model baseline with task-coalition decoding adapted to CHRSP.
- **HDRL**: preserves history-aware dispatching and route-context modeling.
- **TDRL**: preserves token-style state coding and GRU-based dynamic token updates.
- **MVMoE**: introduces sparse mixture-of-experts layers into the AM-style architecture.
- **ECHO**: preserves dual-modality task encoding and historical-resource-aware decoding.

Examples:

```bash
python scripts/eval_drl.py \
  --method am \
  --robot_type 2 \
  --dataset Synthetic_Dataset \
  --model checkpoints/am/type_2/size_20 \
  --decode_strategy greedy \
  --eval_batch_size 1

python scripts/eval_drl.py \
  --method mvmoe \
  --robot_type 3 \
  --dataset Synthetic_Dataset \
  --model checkpoints/mvmoe/type_3/size_50 \
  --decode_strategy sample \
  --width 1280 \
  --eval_batch_size 1
```

## Train DRL Models

Train Co-HeT:

```bash
python scripts/train_drl.py \
  --method cohet \
  --robot_type 2 \
  --graph_size 20 \
  --run_name cohet_type2_20
```

Train adapted DRL baselines:

```bash
python scripts/train_drl.py --method am    --robot_type 2 --graph_size 20 --run_name am_type2_20
python scripts/train_drl.py --method hdrl  --robot_type 2 --graph_size 20 --run_name hdrl_type2_20
python scripts/train_drl.py --method tdrl  --robot_type 2 --graph_size 20 --run_name tdrl_type2_20
python scripts/train_drl.py --method mvmoe --robot_type 2 --graph_size 20 --run_name mvmoe_type2_20
python scripts/train_drl.py --method echo  --robot_type 2 --graph_size 20 --run_name echo_type2_20
```

Additional training arguments are forwarded to the method-specific `run.py`. For a lightweight smoke test:

```bash
python scripts/train_drl.py \
  --method cohet \
  --robot_type 2 \
  --graph_size 20 \
  --n_epochs 1 \
  --epoch_size 512 \
  --batch_size 128 \
  --run_name smoke_cohet
```

For larger instances, encoder checkpointing can reduce GPU memory usage:

```bash
python scripts/train_drl.py \
  --method cohet \
  --robot_type 3 \
  --graph_size 100 \
  --checkpoint_encoder \
  --run_name cohet_type3_100
```

## Run Conventional Baselines

The repository includes exact and metaheuristic baselines:

- `gurobi`: Gurobi MIP model.
- `alns`: Adaptive Large Neighborhood Search.
- `iga`: Iterated Greedy Algorithm.
- `dabc`: Discrete Artificial Bee Colony.
- `diwo`: Discrete Invasive Weed Optimization.

Example:

```bash
python scripts/run_conventional.py \
  --solver alns \
  N20_K2_M12_I1
```

Metaheuristic settings and operator details are summarized in `docs/metaheuristic_baselines.md`.

## Batch Benchmark

Run a batch benchmark over Co-HeT and DRL baselines:

```bash
python scripts/benchmark_all.py \
  --dataset Synthetic_Dataset \
  --methods cohet am hdrl tdrl mvmoe echo \
  --robot_types 2 3 \
  --decode_strategies greedy sample \
  --sample_width 1280 \
  --eval_batch_size 1 \
  --sizes 10 20 50 100 \
  --out_prefix results/comparison/final_synthetic
```

The script writes:

```text
results/comparison/final_synthetic.csv
```

## Reproducibility Checks

Compile all Python sources:

```bash
python -m compileall methods scripts
```

Run a minimal CPU checkpoint loading test:

```bash
python scripts/eval_drl.py \
  --method cohet \
  --robot_type 2 \
  --dataset Synthetic_Dataset \
  --model checkpoints/cohet/type_2/size_20 \
  --decode_strategy greedy \
  --eval_batch_size 1 \
  --val_size 1 \
  --no_cuda
```

## Citation

If this repository helps your research, please cite the corresponding paper.
