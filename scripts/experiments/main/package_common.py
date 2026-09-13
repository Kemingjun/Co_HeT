import json
import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
METHODS = ("Co-HeT", "AM", "MVMoE", "HDRL", "TDRL", "ECHO")
METHOD_SLUGS = {
    "Co-HeT": "cohet",
    "AM": "am",
    "MVMoE": "mvmoe",
    "HDRL": "hdrl",
    "TDRL": "tdrl",
    "ECHO": "echo",
}
KAPPAS = (2, 3)
SIZES = (10, 20, 50, 100)
MODES = ("greedy", "sample1280")
SAMPLE_WIDTHS = {"greedy": 0, "sample1280": 1280}
ROBOT_NUM_LISTS = {2: (4, 8), 3: (4, 6, 8)}
EVAL_SEED = 20260827
TRAINING_SEED = 1234
TIMING_REPEATS = 5





def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def method_slug(method):
    try:
        return METHOD_SLUGS[method]
    except KeyError as error:
        raise ValueError("Unknown method: {}".format(method)) from error


def numeric_instance_index(path_or_name):
    name = Path(path_or_name).name
    match = re.search(r"_I(\d+)\.xlsx$", name)
    if match is None:
        raise ValueError("Unexpected instance filename: {}".format(name))
    return int(match.group(1))


def cell_name(kappa, size):
    if kappa not in KAPPAS or size not in SIZES:
        raise ValueError("Invalid cell: kappa={}, size={}".format(kappa, size))
    return "N{}_K{}_M{}".format(size, kappa, sum(ROBOT_NUM_LISTS[kappa]))


def task_id(method, kappa, size, mode):
    if method not in METHODS or kappa not in KAPPAS or size not in SIZES or mode not in MODES:
        raise ValueError(
            "Invalid task: method={}, kappa={}, size={}, mode={}".format(method, kappa, size, mode)
        )
    return "{}_k{}_n{}_{}".format(method_slug(method), kappa, size, mode)


def iter_tasks(methods=METHODS, kappas=KAPPAS, sizes=SIZES, modes=MODES):
    for method in methods:
        for kappa in kappas:
            for size in sizes:
                for mode in modes:
                    yield {
                        "method": method,
                        "kappa": kappa,
                        "size": size,
                        "mode": mode,
                        "width": SAMPLE_WIDTHS[mode],
                        "task_id": task_id(method, kappa, size, mode),
                    }


def validate_run_id(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run-id must match [A-Za-z0-9][A-Za-z0-9._-]*")
    return run_id


def package_relative(path):
    return Path(path).resolve().relative_to(PACKAGE_ROOT).as_posix()


def resolve_package_relative(path):
    resolved = (PACKAGE_ROOT / path).resolve()
    resolved.relative_to(PACKAGE_ROOT)
    return resolved

