from __future__ import annotations

import datetime
import json
import math
import os
import platform
import random
import socket
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from run_support import CONFIG, INSTANCE_START, INSTANCE_END, dataset_identity

ROOT = Path(__file__).resolve().parent
METHODS = tuple(CONFIG['methods'])
SIZES = tuple(CONFIG['sizes'])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))








def evaluation_cells():
    return [(m, n, i) for m in METHODS for n in SIZES for i in range(INSTANCE_START, INSTANCE_END + 1)]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_strict(model, path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload.get('model', payload)
    clean = {k.removeprefix('module.'): v for k, v in state.items()}
    if len(clean) != len(state):
        raise ValueError('Duplicate checkpoint keys after module prefix removal')
    model.load_state_dict(clean, strict=True)


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.pending-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def validate_run(root, config, resume):
    path = Path(root) / 'config.json'
    if path.exists():
        if read_json(path) != config:
            raise ValueError('Configuration conflict for existing run')
        if not resume:
            raise FileExistsError('Existing run requires --resume')
    else:
        write_once(path, config)


def next_attempt(root, instance, resume):
    paths = sorted((Path(root) / instance).glob('attempt-*.json'))
    for path in paths:
        row = read_json(path)
        if row.get('status') == 'ok':
            if not resume:
                raise FileExistsError('Successful record already exists')
            return path, True
    if paths and not resume:
        raise FileExistsError('Existing attempts require --resume')
    return Path(root) / instance / f'attempt-{len(paths) + 1:03d}.json', False


def validate_records(records):
    actual = [(r['method'], r['size'], r['instance_index']) for r in records]
    expected = set(evaluation_cells())
    identity = dataset_identity()
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f'Expected exactly {len(expected)} unique formal evaluation cells')
    for r in records:
        if any(r.get(k) != v for k, v in identity.items()):
            raise ValueError('Mixed dataset identities')
        if r['status'] != 'ok':
            raise ValueError('Non-successful formal record')
        for key in ('objective', 'travel_time', 'tardiness', 'runtime_s', 'objective_replay_error'):
            if not math.isfinite(r[key]) or r[key] < 0:
                raise ValueError(f'Invalid {key}')
        if r['objective_replay_error'] > 1e-6 or abs(r['objective'] - .5 * (r['travel_time'] + r['tardiness'])) > 1e-6:
            raise ValueError('Objective consistency violation')


def environment():
    versions = {}
    from importlib.metadata import version, PackageNotFoundError
    for name in ('numpy', 'pandas', 'openpyxl', 'torch', 'scipy', 'pytest'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                          '--format=csv,noheader'], capture_output=True, text=True, check=False).stdout if torch.cuda.is_available() else ''
    return dict(hostname=socket.gethostname(), pid=os.getpid(), python=platform.python_version(),
                dependencies=versions, cuda=torch.version.cuda, gpu_inventory=gpu,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), timestamp=now())
