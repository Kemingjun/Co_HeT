import argparse
import random
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Util.RealWorldConfig import RealWorldConfig


INSTANCE_COLUMNS = [
    RealWorldConfig.TASK_INDEX_COL,
    RealWorldConfig.SUPPLY_X_COL,
    RealWorldConfig.SUPPLY_Y_COL,
    RealWorldConfig.HANDOVER_X_COL,
    RealWorldConfig.HANDOVER_Y_COL,
    RealWorldConfig.DELIVERY_X_COL,
    RealWorldConfig.DELIVERY_Y_COL,
    RealWorldConfig.DEADLINE_COL,
]


def generate_deadlines(size, rng):
    deadlines = []
    for task_index in range(1, size + 1):
        noise = int(rng.uniform(-1, 1) * RealWorldConfig.DEADLINE_NOISE)
        deadline = (
            RealWorldConfig.DEADLINE_BASE
            + task_index * RealWorldConfig.DEADLINE_STEP
            + noise
        )
        deadlines.append(deadline)
    rng.shuffle(deadlines)
    return deadlines


def generate_instance(size, seed=None):
    rng = random.Random(seed)
    deadlines = generate_deadlines(size, rng)
    rows = []

    for task_index in range(1, size + 1):
        is_left = rng.randint(0, 1) == 0
        supply_y = rng.choice(RealWorldConfig.STATION_Y_SLOTS)
        handover_y = rng.choice(RealWorldConfig.STATION_Y_SLOTS)

        if is_left:
            supply_x = RealWorldConfig.LEFT_SUPPLY_X
            handover_x = RealWorldConfig.LEFT_HANDOVER_X
        else:
            supply_x = RealWorldConfig.RIGHT_SUPPLY_X
            handover_x = RealWorldConfig.RIGHT_HANDOVER_X

        delivery_x = rng.choice(RealWorldConfig.DELIVERY_X_SLOTS)
        delivery_y = rng.choice(RealWorldConfig.STATION_Y_SLOTS)

        rows.append(
            [
                task_index,
                supply_x,
                supply_y,
                handover_x,
                handover_y,
                delivery_x,
                delivery_y,
                deadlines[task_index - 1],
            ]
        )

    return pd.DataFrame(rows, columns=INSTANCE_COLUMNS)


def save_instance_excel(instance_df, output_dir, filename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    instance_df.to_excel(output_path, index=False)
    return output_path


def generate_batch(output_dir=None, sizes=None, count=None, seed=20260522):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "Instance"
    else:
        output_dir = Path(output_dir)
    if sizes is None:
        sizes = RealWorldConfig.REAL_WORLD_INSTANCE_SIZES
    if count is None:
        count = RealWorldConfig.REAL_WORLD_INSTANCE_COUNT

    generated_paths = []
    for size in sizes:
        for instance_index in range(1, count + 1):
            instance_seed = seed + size * 1000 + instance_index
            instance_df = generate_instance(size, seed=instance_seed)
            filename = f"RW_N{size}_K{RealWorldConfig.ROBOT_TYPE_NUM}_M{RealWorldConfig.ROBOT_NUM}_I{instance_index}.xlsx"
            generated_paths.append(
                save_instance_excel(instance_df, output_dir, filename)
            )
    return generated_paths


def plot_layout(instance_df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    main_width = RealWorldConfig.MAIN_X_MAX - RealWorldConfig.MAIN_X_MIN
    main_height = RealWorldConfig.MAIN_Y_MAX - RealWorldConfig.MAIN_Y_MIN
    ax.add_patch(
        Rectangle(
            (RealWorldConfig.MAIN_X_MIN, RealWorldConfig.MAIN_Y_MIN),
            main_width,
            main_height,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
            label="Main workshop area",
        )
    )

    ax.scatter(
        instance_df[RealWorldConfig.SUPPLY_X_COL],
        instance_df[RealWorldConfig.SUPPLY_Y_COL],
        c="#1f77b4",
        marker="s",
        s=70,
        label=RealWorldConfig.SUPPLY_BUFFER_NAME,
    )
    ax.scatter(
        instance_df[RealWorldConfig.HANDOVER_X_COL],
        instance_df[RealWorldConfig.HANDOVER_Y_COL],
        c="#d62728",
        marker="s",
        s=70,
        label=RealWorldConfig.HANDOVER_STATION_NAME,
    )
    ax.scatter(
        instance_df[RealWorldConfig.DELIVERY_X_COL],
        instance_df[RealWorldConfig.DELIVERY_Y_COL],
        c="#7f7f7f",
        marker="s",
        s=70,
        label=RealWorldConfig.DELIVERY_STATION_NAME,
    )

    ax.axvline(RealWorldConfig.LEFT_SUPPLY_X, color="#1f77b4", linestyle="--", linewidth=0.8)
    ax.axvline(RealWorldConfig.RIGHT_SUPPLY_X, color="#1f77b4", linestyle="--", linewidth=0.8)
    ax.axvline(RealWorldConfig.LEFT_HANDOVER_X, color="#d62728", linestyle=":", linewidth=0.8)
    ax.axvline(RealWorldConfig.RIGHT_HANDOVER_X, color="#d62728", linestyle=":", linewidth=0.8)

    ax.set_xlim(RealWorldConfig.LEFT_SUPPLY_X - 10, RealWorldConfig.RIGHT_SUPPLY_X + 10)
    ax.set_ylim(RealWorldConfig.MAIN_Y_MIN - 5, RealWorldConfig.MAIN_Y_MAX + 5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Industrial Simulation Layout")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate industrial simulation instances.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sizes", nargs="*", type=int, default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--plot-sample", action="store_true")
    parser.add_argument("--plot-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    generated_paths = generate_batch(
        output_dir=args.output_dir,
        sizes=args.sizes,
        count=args.count,
        seed=args.seed,
    )
    if args.plot_sample:
        sample_df = pd.read_excel(generated_paths[0])
        if args.plot_path is None:
            plot_path = Path(__file__).resolve().parent.parent / "result" / "real_world_layout_sample.png"
        else:
            plot_path = Path(args.plot_path)
        plot_layout(sample_df, plot_path)
    print(f"Generated {len(generated_paths)} industrial simulation instance files.")


if __name__ == "__main__":
    main()
