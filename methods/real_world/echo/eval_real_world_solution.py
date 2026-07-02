import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONVENTIONAL_ROOT = ROOT.parent / "real-world-ALNS-Gurobi"
if str(CONVENTIONAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONVENTIONAL_ROOT))

from nets.attention_model import AttentionModel  # noqa: E402
from problems.hrsp.problem_hrsp import HRSP, HRSPDataset  # noqa: E402
from Util.load_data import read_excel  # noqa: E402
from Util.solution_convert import pi_to_path_map, path_map_to_solution  # noqa: E402


def evaluate_instance(instance_name, checkpoint=None, embedding_dim=128, hidden_dim=128, n_encode_layers=1, n_heads=8):
    dataset = HRSPDataset(filename=instance_name)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    model = AttentionModel(
        embedding_dim,
        hidden_dim,
        HRSP,
        n_encode_layers=n_encode_layers,
        n_heads=n_heads,
    )
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict({**model.state_dict(), **payload.get("model", payload)})
    model.set_decode_type("greedy")
    model.eval()

    start = time.time()
    with torch.no_grad():
        raw_cost, _, pi = model(batch, return_pi=True)
    runtime = time.time() - start

    path_map = pi_to_path_map(pi[0])
    instance = read_excel(Path(instance_name).name)
    solution = path_map_to_solution(instance, path_map)
    evaluator_cost = solution.get_fitness()

    return {
        "instance": Path(instance_name).stem,
        "raw_cost": float(raw_cost.squeeze().item()),
        "raw_distance": float(solution.distance),
        "raw_tardiness": float(solution.tardiness),
        "evaluator_cost": float(evaluator_cost),
        "evaluator_distance": float(solution.distance),
        "evaluator_tardiness": float(solution.tardiness),
        "runtime": runtime,
        "path_map": repr(path_map),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="+")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_encode_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=8)
    args = parser.parse_args()

    rows = [
        evaluate_instance(
            instance,
            checkpoint=args.checkpoint,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            n_encode_layers=args.n_encode_layers,
            n_heads=args.n_heads,
        )
        for instance in args.instances
    ]
    for row in rows:
        print(row)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()


