import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation_support import ROOT, METHODS, SIZES, read_json, validate_records
from run_support import INSTANCE_COUNT, INSTANCE_DIR, dataset_identity, validate_instance


def collect(raw_root):
    records, failures = [], []
    identity = dataset_identity()
    for p in sorted(Path(raw_root).rglob('attempt-*.json')):
        row = read_json(p)
        config = read_json(p.parent.parent / 'config.json')
        if any(row.get(k) != v or config.get(k) != v for k, v in identity.items()):
            raise ValueError(f'Mixed or incorrect dataset identity: {p}')
        entry = validate_instance(INSTANCE_DIR / row['instance'])
        if row['size'] != entry['size'] or row['instance_index'] != entry['index']:
            raise ValueError(f'Instance identity mismatch: {p}')
        if any(row[k] != config[k] for k in ('method', 'size', 'eval_seed')):
            raise ValueError(f'Result configuration mismatch: {p}')
        if not config['instance_start'] <= row['instance_index'] <= config['instance_end']:
            raise ValueError(f'Instance outside selected configuration: {p}')
        if row['status'] != 'ok':
            failures.append(row)
            continue
        if config['kind'] != 'drl_evaluation' or row['decode'] != 'sample-1280' or row['eval_seed'] != 20260825:
            raise ValueError('Non-formal record included')
        import math
        errors = row['component_replay_errors']
        if not row['feasible'] or set(errors) != {'travel_distance', 'travel_time', 'tardiness'} or any(not math.isfinite(v) or v < 0 or v > 1e-6 for v in errors.values()):
            raise ValueError('Infeasible or invalid component replay')
        records.append(row)
    validate_records(records)
    order = {m: i for i, m in enumerate(METHODS)}
    return sorted(records, key=lambda r: (order[r['method']], r['size'], r['instance_index'])), failures


def csv_once(path, rows, fields):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                          for k, v in row.items()} for row in rows)


def summarize(raw_root, output_dir):
    records, failures = collect(raw_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260825)
    # Resample instances, using the same indices across methods and metrics at each size.
    indices = {n: rng.integers(0, INSTANCE_COUNT, size=(10000, INSTANCE_COUNT)) for n in SIZES}
    rows = []
    metrics = ('objective', 'travel_time', 'tardiness', 'runtime_s')
    for m in METHODS:
        for n in SIZES:
            cell = [r for r in records if r['method'] == m and r['size'] == n]
            row = dict(**dataset_identity(), method=m, size=n, count=len(cell))
            for key in metrics:
                x = np.array([r[key] for r in cell], dtype=np.float64)
                low, high = np.quantile(x[indices[n]].mean(axis=1), [.025, .975])
                row.update({key + '_mean': float(x.mean()), key + '_sd': float(x.std(ddof=1)),
                            key + '_ci95_low': float(low), key + '_ci95_high': float(high)})
            rows.append(row)
    fields = ['dataset_id', 'method', 'size', 'instance', 'instance_index', 'training_seed', 'eval_seed', 'decode',
              'objective', 'travel_time', 'tardiness', 'travel_distance', 'runtime_s',
              'objective_replay_error', 'peak_gpu_memory_bytes', 'status', 'error',
              'raw_model_objective', 'feasible', 'component_replay_errors', 'action_sequence', 'path_map', 'task_timeline']
    csv_once(output / 'per_instance_results.csv', records + failures, fields)
    csv_once(output / 'method_size_summary.csv', rows, list(rows[0]))
    labels = {'objective': 'Obj.', 'travel_time': 'Trav. time', 'tardiness': 'Tard.', 'runtime_s': 'Time'}
    with (output / 'drl_table_fragment.tex').open('x', encoding='utf-8') as stream:
        stream.write('% Columns: n, metric, AM, MVMoE, HDRL, TDRL, ECHO, Co-HeT\n')
        stream.write('% No RPD or RPD SD: merge audited traditional results before computing BKS.\n')
        for n in SIZES:
            for key in metrics:
                values = [next(r for r in rows if r['method'] == m and r['size'] == n)[key + '_mean']
                          for m in ('am', 'mvmoe', 'hdrl', 'tdrl', 'echo', 'cohet')]
                digits = 3 if key == 'runtime_s' else 1
                stream.write(f'{n} & {labels[key]} & ' + ' & '.join(f'{v:.{digits}f}' for v in values) + ' \\\\\n')
    print(f'PASS: {len(records)} records, {len(rows)} complete cells; summary at {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'results/raw/drl')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.raw_root, args.output_dir)
