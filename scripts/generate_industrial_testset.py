"""Create the independent 400-instance industrial simulation test set."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'scripts/data_sources/industrial'
DATASET_ID = 'real_world_test100_seed20260906'
OUTPUT_DIR = PROJECT / 'instances' / DATASET_ID
RECEIPT_DIR = PROJECT / 'outputs/instance_generation'
SIZES = (10, 20, 30, 40)
MASTER_SEED = 20260906
COLUMNS = ['task_index', 'supply_x', 'supply_y', 'handover_x', 'handover_y',
           'delivery_x', 'delivery_y', 'deadline']
GENERATION_CONFIG = {
    'ROBOT_TYPE_NUM': 3, 'ROBOT_NUM': 16, 'ROBOT_NUM_LIST': [4, 8, 4],
    'LEFT_SUPPLY_X': -20, 'RIGHT_SUPPLY_X': 120,
    'LEFT_HANDOVER_X': 0, 'RIGHT_HANDOVER_X': 100,
    'STATION_Y_SLOTS': list(range(5, 100, 5)), 'DELIVERY_X_SLOTS': [20, 40, 60, 80],
    'DEADLINE_BASE': 300, 'DEADLINE_STEP': 40, 'DEADLINE_NOISE': 40,
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_generator():
    generator_path = SOURCE / 'real_world_instance_generator.py'
    config_path = SOURCE / 'RealWorldConfig.py'
    config_module = load_module('_industrial_original_config', config_path)
    for key, expected in GENERATION_CONFIG.items():
        if getattr(config_module.RealWorldConfig, key) != expected:
            raise ValueError(f'Original generation configuration changed: {key}')
    # Isolate the original Util import from model packages already imported by tests.
    previous = {key: sys.modules.get(key) for key in ('Util', 'Util.RealWorldConfig')}
    old_path = list(sys.path)
    try:
        package = types.ModuleType('Util')
        package.__path__ = [str(SOURCE)]
        sys.modules['Util'] = package
        sys.modules['Util.RealWorldConfig'] = config_module
        generator = load_module('_industrial_original_generator', generator_path)
    finally:
        sys.path[:] = old_path
        for key, module in previous.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module
    if generator.INSTANCE_COLUMNS != COLUMNS:
        raise ValueError('Original instance columns changed')
    return generator.generate_instance


def cells():
    return ((n, i) for n in SIZES for i in range(1, 101))


def filename(n, index):
    return f'RW_N{n}_K3_M16_I{index}.xlsx'


def instance_seed(n, index):
    if n not in SIZES or not 1 <= index <= 100:
        raise ValueError('Unsupported size or instance index')
    return MASTER_SEED + n * 1000 + index


def validate_frame(frame, n, seed):
    if list(frame.columns) != COLUMNS or frame.shape != (n, 8):
        raise ValueError('Invalid columns or row count')
    for row in frame.itertuples(index=False, name=None):
        for value in row:
            number = float(value)
            if not math.isfinite(number) or number != int(number):
                raise ValueError('Expected finite integer-valued instance cells')
    if frame.task_index.tolist() != list(range(1, n + 1)):
        raise ValueError('Invalid task order')
    for row in frame.itertuples(index=False):
        if (row.supply_x, row.handover_x) not in ((-20, 0), (120, 100)):
            raise ValueError('Invalid same-side supply and handover positions')
        if row.delivery_x not in (20, 40, 60, 80):
            raise ValueError('Invalid delivery x slot')
        if any(y not in range(5, 100, 5) for y in (row.supply_y, row.handover_y, row.delivery_y)):
            raise ValueError('Invalid station y slot')
    rng = random.Random(seed)
    deadlines = [300 + 40 * t + int(rng.uniform(-1, 1) * 40) for t in range(1, n + 1)]
    rng.shuffle(deadlines)
    if frame.deadline.tolist() != deadlines:
        raise ValueError('Deadline sequence does not match original generation rule')


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def generation_config():
    return {
        'dataset_id': DATASET_ID,
        'sizes': list(SIZES),
        'count_per_size': 100,
        'master_seed': MASTER_SEED,
        'seed_formula': '20260906 + size * 1000 + index',
        'generation_config': GENERATION_CONFIG,
        'columns': COLUMNS,
    }


def dataset_readme():
    return (
        '# Independent industrial simulation test set\n\n'
        f'Dataset ID: `{DATASET_ID}`. Four sizes (10, 20, 30, 40), 100 instances each.\n'
        'Names repeat across datasets: identify instances by dataset ID, size, and index.\n\n'
        'Generation uses the generate_instance function with master seed 20260906.\n'
        'Per-instance seed = 20260906 + size * 1000 + index. See generation_config.json for parameters.\n'
        'Left/right sides are equiprobable. Supply and handover share a side; y slots are 5:5:95.\n'
        'Supply x: -20/120; handover x: 0/100; delivery x: 20/40/60/80.\n'
        'Deadline = 300 + 40 * task_index + int(Uniform[-1,1] * 40), followed by shuffle.\n'
        'The integer conversion truncates toward zero.\n\n'
        'Coordinates are in meters; deadlines are in seconds. These simulation test instances\n'
        'are generated using parameters collected from the industrial site.\n'
        'Evaluation uses Manhattan distances, open paths, and 0.5 travel_time + 0.5 tardiness,\n'
        'with carrier speed 1.2 and forklift speed 1.0; shuttle travel is not double-counted.\n'
        '\nThe generator requires a new output directory.\n'
    )


def validate_dataset(output):
    """Check required workbook count and schema."""
    output = Path(output)
    if sum(1 for _ in output.glob('*.xlsx')) != len(SIZES) * 100:
        raise ValueError('Expected 400 instance workbooks')
    for n, index in cells():
        frame = pd.read_excel(output / filename(n, index), engine='openpyxl')
        validate_frame(frame, n, instance_seed(n, index))


def run(mode, output=OUTPUT_DIR, receipts=RECEIPT_DIR):
    output, receipts = Path(output), Path(receipts)
    if mode not in ('dry-run', 'write'):
        raise ValueError('Unknown mode')
    if mode == 'write' and output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    generate = load_generator()
    frames = {}
    for n, index in cells():
        seed = instance_seed(n, index)
        frame = generate(n, seed=seed)
        validate_frame(frame, n, seed)
        frames[n, index] = frame
    if mode == 'dry-run':
        return {'status': 'PASS', 'mode': mode, 'count': len(frames), 'files_written': 0,
                'dataset_id': DATASET_ID, 'output_dir': str(output), 'target_exists': output.exists()}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{DATASET_ID}-staging-', dir=output.parent))
    print(f'Generating 400 instances in {staging}', flush=True)
    for (n, index), frame in frames.items():
        frame.to_excel(staging / filename(n, index), index=False, engine='openpyxl')
    (staging / 'README.md').write_text(dataset_readme(), encoding='utf-8')
    write_json(staging / 'generation_config.json', generation_config())
    validate_dataset(staging)
    if output.exists():
        raise FileExistsError(f'Target appeared during generation; staging retained: {staging}')
    os.rename(staging, output)
    result = {'status': 'PASS', 'count': len(frames), 'dataset_id': DATASET_ID,
              'output_dir': str(output)}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    receipt_dir = receipts / f'{DATASET_ID}_{stamp}'
    receipt_dir.mkdir(parents=True, exist_ok=False)
    write_json(receipt_dir / 'receipt.json', dict(result, created_utc=stamp, command=sys.argv))
    result['receipt'] = str(receipt_dir / 'receipt.json')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--write', action='store_true')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    mode = 'write' if args.write else 'dry-run'
    try:
        print(json.dumps(run(mode), indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
