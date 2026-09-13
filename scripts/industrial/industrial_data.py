"""Select and validate industrial instances by dataset, scale and instance number."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_DIR = ROOT / 'instances/real_world_test100_seed20260906'
DATASET_ID = 'real_world_test100_seed20260906'
ALLOWED_SIZES = (10, 20, 30, 40)
COLUMNS = ('task_index', 'supply_x', 'supply_y', 'handover_x', 'handover_y',
           'delivery_x', 'delivery_y', 'deadline')

def dataset_identity():
    config = json.loads((INSTANCE_DIR / 'generation_config.json').read_text(encoding='utf-8-sig'))
    if config['dataset_id'] != DATASET_ID or config['sizes'] != list(ALLOWED_SIZES) or config['count_per_size'] != 100:
        raise ValueError('Industrial dataset configuration does not match the test matrix')
    return {'dataset_id': DATASET_ID}

@lru_cache(maxsize=400)
def validate_instance(path):
    from openpyxl import load_workbook
    path = Path(path)
    import re
    match = re.fullmatch(r'RW_N(10|20|30|40)_K3_M16_I(\d+)\.xlsx', path.name)
    if not match or not 1 <= int(match[2]) <= 100 or not path.is_file():
        raise ValueError(f'Invalid or missing industrial instance: {path}')
    n, index = int(match[1]), int(match[2])
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook.active.values)
    finally:
        workbook.close()
    if not rows or tuple(rows[0]) != COLUMNS or len(rows) != n + 1:
        raise ValueError(f'Unexpected columns or task count: {path}')
    if [row[0] for row in rows[1:]] != list(range(1, n + 1)):
        raise ValueError(f'Missing or duplicate task indices: {path}')
    return {'filename': path.name, 'size': n, 'index': index, 'seed': 20260906 + n * 1000 + index}

def select_instances(*, sizes, instance_start, instance_end):
    sizes = list(sizes)
    if not sizes or len(sizes) != len(set(sizes)) or not set(sizes).issubset(ALLOWED_SIZES):
        raise ValueError(f'Unsupported or duplicated industrial sizes: {sizes}')
    if not 1 <= instance_start <= instance_end <= 100:
        raise ValueError('Industrial instance indices must be 1 through 100')
    dataset_identity()
    paths = [INSTANCE_DIR / f'RW_N{n}_K3_M16_I{i}.xlsx'
             for n in sorted(sizes) for i in range(instance_start, instance_end + 1)]
    for path in paths:
        validate_instance(path)
    return paths
