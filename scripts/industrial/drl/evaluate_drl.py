from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import sys
import time
import traceback
import os
import logging
from pathlib import Path

import torch
from evaluation_support import (load_checkpoint_strict, next_attempt, validate_run, write_once,
                                seed_everything, environment, now, read_json)

from run_support import (
    PACKAGE_ROOT,
    load_experiment_config,
    select_instances,
    write_json_once,
    write_or_validate_config,
    CONFIG, INSTANCE_START, INSTANCE_END, ALLOWED_SIZES, dataset_identity, validate_instance,
)


METHODS = tuple(CONFIG['methods'])


def _activate_method(method: str):
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    method_dir = Path(__file__).resolve().parents[3] / "methods/real_world" / method
    conventional_dir = Path(__file__).resolve().parents[3] / "methods/conventional/industrial/replay"
    for path in (str(method_dir), str(conventional_dir)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    attention_module = importlib.import_module("nets.attention_model")
    problem_module = importlib.import_module("problems.hrsp.problem_hrsp")
    solution_module = importlib.import_module("Util.solution_convert")
    return attention_module.AttentionModel, problem_module, solution_module


def _instantiate_model(model_cls, problem_cls, options: dict):
    candidates = {
        "embedding_dim": int(options.get("embedding_dim", 128)),
        "hidden_dim": int(options.get("hidden_dim", 128)),
        "problem": problem_cls,
        "n_encode_layers": int(options.get("n_encode_layers", 1)),
        "tanh_clipping": float(options.get("tanh_clipping", 10.0)),
        "normalization": options.get("normalization", "batch"),
        "checkpoint_encoder": bool(options.get("checkpoint_encoder", False)),
        "shrink_size": options.get("shrink_size"),
        "n_heads": int(options.get("n_heads", 8)),
        "use_moe": bool(options.get("use_moe", True)),
        "moe_num_experts": int(options.get("moe_num_experts", 4)),
        "moe_top_k": int(options.get("moe_top_k", 2)),
        "moe_position": options.get("moe_position", "encoder_decoder"),
        "use_echo_encoder": bool(options.get("use_echo_encoder", True)),
        "use_pfca_context": bool(options.get("use_pfca_context", True)),
        "echo_edge_dim": int(options.get("echo_edge_dim", 128)),
    }
    signature = inspect.signature(model_cls)
    accepted = {name: value for name, value in candidates.items() if name in signature.parameters}
    return model_cls(**accepted)


def _load_checkpoint(model, checkpoint: Path, device: torch.device) -> None:
    load_checkpoint_strict(model, checkpoint)


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _normalize_pi(pi):
    if torch.is_tensor(pi) and pi.dim() == 1:
        if pi.numel() % 4 != 0:
            raise ValueError("Flattened industrial action sequence is not divisible by four")
        return pi.view(-1, 4)
    return pi


def _evaluate_instance(args, model, problem_module, solution_module, instance_path, device):
    entry = validate_instance(instance_path)
    seed_everything(args.eval_seed)
    sample = problem_module.load_excel_file(instance_path)
    batch = _to_device({key: value.unsqueeze(0) for key, value in sample.items()}, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started_utc = now()
    started = time.perf_counter()
    with torch.no_grad():
        model.set_decode_type("sampling")
        pi, raw_cost = model.sample_many(batch, batch_rep=args.sample_width, iter_rep=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started

    selected_pi = _normalize_pi(pi[0])
    if selected_pi.shape != (args.size, 4):
        raise ValueError(f'Unexpected action shape: {selected_pi.shape}')
    if sorted(selected_pi[:, 0].detach().cpu().tolist()) != list(range(args.size)):
        raise ValueError('Action sequence must cover every task exactly once')
    path_map = solution_module.pi_to_path_map(selected_pi)
    from Util.load_data import read_excel
    conventional_instance = read_excel(str(instance_path.resolve()))
    solution = solution_module.path_map_to_solution(conventional_instance, path_map)
    replay_objective = float(solution.get_fitness())
    if not solution.feasible:
        raise ValueError('Shared schedule replay reports an infeasible solution')
    # Replay the selected schedule through the method state independently of selection.
    state = problem_module.HRSP.make_state(batch)
    for action in selected_pi:
        chosen = torch.zeros((1, 16), dtype=torch.bool, device=device)
        chosen[0, action[1:].long()] = True
        state = state.update(action[:1].long(), chosen)
    component_errors = {
        'travel_distance': abs(float(state.travel_distance.item()) - float(solution.distance)),
        'travel_time': abs(float(state.travel_time.item()) - float(solution.travel_time)),
        'tardiness': abs(float(state.tardiness.item()) - float(solution.tardiness)),
    }
    if any(not math.isfinite(v) or v > 1e-6 for v in component_errors.values()):
        raise RuntimeError(f'Component replay mismatch: {component_errors}')
    model_objective = float(raw_cost.reshape(-1)[0].detach().cpu().item())
    replay_error = abs(model_objective - replay_objective)
    if not math.isfinite(replay_error) or replay_error > 1e-6:
        raise RuntimeError(f"Objective replay error {replay_error:.12g} exceeds 1e-6")
    index = int(instance_path.stem.rsplit("_I", 1)[1])
    return {
        **args.dataset_identity,
        "method": args.method,
        "size": args.size,
        "instance": instance_path.name,
        "instance_index": index,
        "training_seed": 1234,
        "eval_seed": args.eval_seed,
        "decode": "sample-1280" if args.sample_width == 1280 else f"diagnostic-sample-{args.sample_width}",
        "travel_distance": float(solution.distance),
        "travel_time": float(solution.travel_time),
        "tardiness": float(solution.tardiness),
        "objective": replay_objective,
        "raw_model_objective": model_objective,
        "objective_replay_error": replay_error,
        "runtime_s": runtime,
        "started_utc": started_utc,
        "finished_utc": now(),
        "pid": os.getpid(),
        "action_sequence": selected_pi.detach().cpu().tolist(),
        "component_replay_errors": component_errors,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
        "feasible": True,
        "path_map": {str(key): value for key, value in solution.get_path_map().items()},
        "task_timeline": {str(key): value for key, value in solution.info_map.items()},
        "status": "ok",
        "error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one retrained industrial DRL checkpoint.")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--size", required=True, type=int, choices=ALLOWED_SIZES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--args-json", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--instance-start", type=int, default=INSTANCE_START)
    parser.add_argument("--instance-end", type=int, default=INSTANCE_END)
    parser.add_argument("--eval-seed", type=int, default=20260825)
    parser.add_argument("--sample-width", type=int, default=1280)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    args.checkpoint = str(checkpoint)
    args_json = Path(args.args_json).resolve() if args.args_json else checkpoint.parent / "args.json"
    args.args_json = str(args_json)
    if not checkpoint.is_file() or not args_json.is_file():
        raise FileNotFoundError("Checkpoint and adjacent args.json are both required")
    if args.eval_seed != 20260825:
        raise ValueError("The formal industrial evaluation seed is locked to 20260825")
    if not args.diagnostic and args.sample_width != 1280:
        raise ValueError("Formal industrial evaluation requires Sample-1280")
    if not args.diagnostic and (args.instance_start, args.instance_end) != (INSTANCE_START, INSTANCE_END):
        raise ValueError(f'Formal evaluations require I{INSTANCE_START}-I{INSTANCE_END}')
    if not args.diagnostic and args.device != 'cuda':
        raise ValueError('Formal timing requires CUDA')
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('run-id must be a single directory name')

    instances = select_instances(sizes=[args.size], instance_start=args.instance_start, instance_end=args.instance_end)
    args.dataset_identity = dataset_identity()
    model_options = read_json(args_json)
    for key, expected in [('graph_size', args.size), ('seed', 1234), ('batch_size', 1024),
                          ('epoch_size', 1280000), ('val_size', 10000)]:
        if model_options.get(key) != expected:
            raise ValueError(f'Training config mismatch: {key}')
    run_config = {
        **args.dataset_identity,
        "kind": "drl_evaluation_diagnostic" if args.diagnostic else "drl_evaluation",
        "method": args.method,
        "size": args.size,
        "checkpoint": str(checkpoint),
        "model_options": model_options,
        "eval_seed": args.eval_seed,
        "sample_width": args.sample_width,
        "instance_start": args.instance_start,
        "instance_end": args.instance_end,
        "device": args.device,
        "timing": "one unchunked Sample-1280 warmup on I1; synchronize before/after; exclude IO and replay",
        "experiment": load_experiment_config(),
    }
    print(f"method={args.method} size={args.size} instances={len(instances)}")
    if args.dry_run:
        return 0

    gpu_lease = None
    gpu_info = None
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this timing protocol")
        gpu_info = {"device": torch.cuda.current_device(), "name": torch.cuda.get_device_name()}
    elif args.diagnostic:
        # Method parameter modules choose their tensor device at import time.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    run_root = PACKAGE_ROOT / ("diagnostics" if args.diagnostic else "results/raw/drl") / args.run_id
    validate_run(run_root, run_config, resume=args.resume)
    device = torch.device(args.device)
    model_cls, problem_module, solution_module = _activate_method(args.method)
    model = _instantiate_model(model_cls, problem_module.HRSP, model_options)
    _load_checkpoint(model, checkpoint, device)
    model.to(device)
    model.eval()

    stamp = now().replace(':', '').replace('.', '')
    env = environment()
    env['command'] = sys.argv
    env['selected_gpu'] = gpu_info
    env.update(args.dataset_identity)
    write_once(run_root / 'sessions' / f'{stamp}.json', env)
    print(f'WARMUP/SMOKE: Sample-{args.sample_width}, excluded from formal records and timing', flush=True)
    try:
        _evaluate_instance(args, model, problem_module, solution_module, instances[0], device)
    except Exception:
        write_once(run_root / 'sessions' / f'{stamp}_warmup_failed.json',
                   {'status': 'failed', 'error': traceback.format_exc(), **args.dataset_identity})
        raise

    failures = 0
    for instance_path in instances:
        record_path, skip = next_attempt(run_root, instance_path.stem, resume=args.resume)
        if skip:
            continue
        try:
            record = _evaluate_instance(args, model, problem_module, solution_module, instance_path, device)
        except Exception as exc:
            failures += 1
            record = {
                **args.dataset_identity,
                "method": args.method,
                "size": args.size,
                "instance": instance_path.name,
                "instance_index": int(instance_path.stem.rsplit('_I', 1)[1]),
                "finished_utc": now(),
                "eval_seed": args.eval_seed,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        record['hostname'] = env['hostname']
        record['gpu_inventory'] = env['gpu_inventory']
        record['selected_gpu'] = gpu_info
        write_once(record_path, record)
        print(f"{instance_path.name}: {record['status']}", flush=True)
        if record['status'] != 'ok':
            print(record['error'], flush=True)
            break
    write_once(run_root / 'sessions' / f'{stamp}_exit.json',
               {'exit_code': 1 if failures else 0, 'finished_utc': now()})
    if gpu_lease is not None:
        gpu_lease.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
