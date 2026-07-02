import os
import pickle
from pathlib import Path


def get_instance_root():
    configured = os.environ.get("COHET_INSTANCE_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent.parent / "Instance"

import torch
from torch.utils.data import Dataset

from problems.hrsp.paramet_hrsp import paramet_hrsp
from problems.hrsp.state_hrsp import StateHRSP


class HRSP(object):
    NAME = "hrsp"

    @staticmethod
    def get_costs(dataset, pi):
        raise NotImplementedError("Cost is computed inside StateHRSP during decoding.")

    @staticmethod
    def make_dataset(*args, **kwargs):
        return HRSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateHRSP.initialize(*args, **kwargs)


class HRSPDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(HRSPDataset, self).__init__()
        if filename is None:
            self.data = make_random_instances(size, num_samples, generator=None)
        else:
            self.data = load_instances(filename, offset, num_samples)
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def normalize_xy(raw_xy):
    result = raw_xy.clone().float()
    result[..., 0] = (result[..., 0] + paramet_hrsp.X_OFFSET) / paramet_hrsp.X_SCALE
    result[..., 1] = result[..., 1] / paramet_hrsp.Y_SCALE
    return result


def make_random_instances(size, num_samples, generator=None):
    side = torch.randint(0, 2, (num_samples, size), generator=generator)
    supply_x = torch.where(
        side == 0,
        torch.full((num_samples, size), paramet_hrsp.LEFT_SUPPLY_X),
        torch.full((num_samples, size), paramet_hrsp.RIGHT_SUPPLY_X),
    )
    handover_x = torch.where(
        side == 0,
        torch.full((num_samples, size), paramet_hrsp.LEFT_HANDOVER_X),
        torch.full((num_samples, size), paramet_hrsp.RIGHT_HANDOVER_X),
    )

    y_indices = torch.randint(
        0,
        len(paramet_hrsp.STATION_Y_SLOTS),
        (num_samples, size, 3),
        generator=generator,
    )
    supply_y = paramet_hrsp.STATION_Y_SLOTS[y_indices[:, :, 0]]
    handover_y = paramet_hrsp.STATION_Y_SLOTS[y_indices[:, :, 1]]
    delivery_y = paramet_hrsp.STATION_Y_SLOTS[y_indices[:, :, 2]]
    delivery_x = paramet_hrsp.DELIVERY_X_SLOTS[
        torch.randint(
            0,
            len(paramet_hrsp.DELIVERY_X_SLOTS),
            (num_samples, size),
            generator=generator,
        )
    ]

    task_order = torch.arange(1, size + 1, dtype=torch.float).unsqueeze(0).expand(num_samples, -1)
    noise = ((torch.rand(num_samples, size, generator=generator) * 2.0 - 1.0) * paramet_hrsp.DEADLINE_NOISE).trunc()
    deadline_unpermuted = paramet_hrsp.DEADLINE_BASE + task_order * paramet_hrsp.DEADLINE_STEP + noise
    permutations = torch.argsort(torch.rand(num_samples, size, generator=generator), dim=1)
    deadline = torch.gather(deadline_unpermuted, 1, permutations)

    supply_raw = torch.stack([supply_x, supply_y], dim=-1)
    handover_raw = torch.stack([handover_x, handover_y], dim=-1)
    delivery_raw = torch.stack([delivery_x, delivery_y], dim=-1)
    supply_norm = normalize_xy(supply_raw)
    handover_norm = normalize_xy(handover_raw)
    delivery_norm = normalize_xy(delivery_raw)
    required_robot = torch.ones(num_samples, size, paramet_hrsp.ROBOT_TYPE_NUM, dtype=torch.float)

    return [
        {
            "supply_raw": supply_raw[i].float(),
            "handover_raw": handover_raw[i].float(),
            "delivery_raw": delivery_raw[i].float(),
            "supply_norm": supply_norm[i].float(),
            "handover_norm": handover_norm[i].float(),
            "delivery_norm": delivery_norm[i].float(),
            "deadline": deadline[i].float(),
            "required_robot": required_robot[i],
        }
        for i in range(num_samples)
    ]


def make_random_instance(size, generator=None):
    side = torch.randint(0, 2, (size,), generator=generator)
    supply_x = torch.where(
        side == 0,
        torch.full((size,), paramet_hrsp.LEFT_SUPPLY_X),
        torch.full((size,), paramet_hrsp.RIGHT_SUPPLY_X),
    )
    handover_x = torch.where(
        side == 0,
        torch.full((size,), paramet_hrsp.LEFT_HANDOVER_X),
        torch.full((size,), paramet_hrsp.RIGHT_HANDOVER_X),
    )
    y_indices = torch.randint(0, len(paramet_hrsp.STATION_Y_SLOTS), (size, 3), generator=generator)
    supply_y = paramet_hrsp.STATION_Y_SLOTS[y_indices[:, 0]]
    handover_y = paramet_hrsp.STATION_Y_SLOTS[y_indices[:, 1]]
    delivery_y = paramet_hrsp.STATION_Y_SLOTS[y_indices[:, 2]]
    delivery_x = paramet_hrsp.DELIVERY_X_SLOTS[
        torch.randint(0, len(paramet_hrsp.DELIVERY_X_SLOTS), (size,), generator=generator)
    ]

    task_order = torch.arange(1, size + 1, dtype=torch.float)
    noise = ((torch.rand(size, generator=generator) * 2.0 - 1.0) * paramet_hrsp.DEADLINE_NOISE).trunc()
    deadline = paramet_hrsp.DEADLINE_BASE + task_order * paramet_hrsp.DEADLINE_STEP + noise
    deadline = deadline[torch.randperm(size, generator=generator)]

    supply_raw = torch.stack([supply_x, supply_y], dim=-1)
    handover_raw = torch.stack([handover_x, handover_y], dim=-1)
    delivery_raw = torch.stack([delivery_x, delivery_y], dim=-1)
    return make_instance_from_tensors(supply_raw, handover_raw, delivery_raw, deadline)


def load_instances(filename, offset, num_samples):
    path = Path(filename)
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            data = pickle.load(f)
        return [make_instance_from_tuple(args) for args in data[offset:offset + num_samples]]

    if path.suffix == ".xlsx":
        return [load_excel_file(resolve_instance_file(path.name))]

    base_dir = get_instance_root() / filename
    files = sorted(base_dir.glob("*.xlsx"))[offset:offset + num_samples]
    return [load_excel_file(file_path) for file_path in files]


def resolve_instance_file(filename):
    local_path = get_instance_root() / filename
    if local_path.exists():
        return local_path
    conventional_path = Path(__file__).resolve().parents[3] / "real-world-ALNS-Gurobi" / "Instance" / filename
    if conventional_path.exists():
        return conventional_path
    raise FileNotFoundError(filename)


def load_excel_file(file_path):
    import pandas as pd

    df = pd.read_excel(file_path)
    required_columns = [
        "supply_x",
        "supply_y",
        "handover_x",
        "handover_y",
        "delivery_x",
        "delivery_y",
        "deadline",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing real-world columns in {file_path}: {missing}")
    supply_raw = torch.tensor(df[["supply_x", "supply_y"]].values, dtype=torch.float)
    handover_raw = torch.tensor(df[["handover_x", "handover_y"]].values, dtype=torch.float)
    delivery_raw = torch.tensor(df[["delivery_x", "delivery_y"]].values, dtype=torch.float)
    deadline = torch.tensor(df["deadline"].values, dtype=torch.float)
    return make_instance_from_tensors(supply_raw, handover_raw, delivery_raw, deadline)


def make_instance_from_tuple(args):
    supply_raw, handover_raw, delivery_raw, deadline, *_ = args
    return make_instance_from_tensors(
        torch.tensor(supply_raw, dtype=torch.float),
        torch.tensor(handover_raw, dtype=torch.float),
        torch.tensor(delivery_raw, dtype=torch.float),
        torch.tensor(deadline, dtype=torch.float),
    )


def make_instance_from_tensors(supply_raw, handover_raw, delivery_raw, deadline):
    required_robot = torch.ones(supply_raw.size(0), paramet_hrsp.ROBOT_TYPE_NUM, dtype=torch.float)
    return {
        "supply_raw": supply_raw.float(),
        "handover_raw": handover_raw.float(),
        "delivery_raw": delivery_raw.float(),
        "supply_norm": normalize_xy(supply_raw),
        "handover_norm": normalize_xy(handover_raw),
        "delivery_norm": normalize_xy(delivery_raw),
        "deadline": deadline.float(),
        "required_robot": required_robot,
    }



