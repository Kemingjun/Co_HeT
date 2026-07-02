import time

from torch.utils.data import Dataset
import torch
import os
import pickle
import ast
from pathlib import Path


def get_instance_root():
    configured = os.environ.get("COHET_INSTANCE_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent.parent / "Instance"
import pandas as pd

# from utils.beam_search import beam_search
from problems.hrsp.state_hrsp import StateHRSP
from problems.hrsp.paramet_hrsp import paramet_hrsp


class HRSP(object):

    NAME = 'hrsp'  # heterogeneous robot scheduling problem


    @staticmethod
    def get_costs(dataset, pi):
        pass

    @staticmethod
    def make_dataset(*args, **kwargs):
        return HRSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateHRSP.initialize(*args, **kwargs)


class HRSPDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(HRSPDataset, self).__init__()

        self.data = []

        self.data_set = []
        if filename is not None:
            if os.path.splitext(filename)[1] == '.pkl':
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                self.data = [make_instance(args) for args in data[offset:offset+num_samples]]
            else:
                if os.path.splitext(filename)[1] == '.xlsx':
                    file_name = str(get_instance_root() / filename)
                    df = pd.read_excel(file_name)
                    instance = [list(row) for index, row in df.iterrows()]
                    for task_info in instance:
                        operation_time = task_info[-2]
                        _operation_time = ast.literal_eval(operation_time)
                        task_info[-2] = _operation_time
                    source = [[row[1], row[2]] for row in instance]
                    deadline = [row[3] for row in instance]
                    operation_time = [row[4] for row in instance]
                    # required_robot = [row[5] for row in instance]

                    source_tensor = torch.tensor(source, dtype=torch.float)
                    deadline_tensor = torch.tensor(deadline, dtype=torch.float)
                    operation_time_tensor = torch.tensor(operation_time, dtype=torch.float)
                    # required_robot_tensor = torch.tensor(required_robot, dtype=torch.long)

                    self.data = [{
                        'source': source_tensor,
                        'deadline': deadline_tensor,
                        # 'required_robot': required_robot_tensor,
                        'operation_time': operation_time_tensor,
                    }]
                else:
                    base_dir = get_instance_root() / filename
                    all_files = list(base_dir.glob("*.xlsx"))
                    for file_path in all_files:
                        df = pd.read_excel(file_path)
                        instance = [list(row) for index, row in df.iterrows()]
                        for task_info in instance:
                            operation_time = task_info[-2]
                            _operation_time = ast.literal_eval(operation_time)
                            task_info[-2] = _operation_time
                        source = [[row[1], row[2]] for row in instance]
                        deadline = [row[3] for row in instance]
                        operation_time = [row[4] for row in instance]
                        # required_robot = [row[5] for row in instance]

                        source_tensor = torch.tensor(source, dtype=torch.float)
                        deadline_tensor = torch.tensor(deadline, dtype=torch.float)
                        operation_time_tensor = torch.tensor(operation_time, dtype=torch.float)
                        self.data.append({
                            'source': source_tensor,
                            'deadline': deadline_tensor,
                            # 'required_robot': required_robot_tensor,
                            'operation_time': operation_time_tensor,
                        })


        else:

            operation_range_list = [[0.3, 0.7], [0.8, 1.2], [0.8, 1.2]]
            self.data = []

            base = torch.arange(size, dtype=torch.float32) * 0.5 + 0.5
            base_batch = base.unsqueeze(0).repeat(num_samples, 1)  # Shape: (num_samples, size)
            noise_batch = (torch.rand(num_samples, size) / 2.5 - 0.2)  # Shape: (num_samples, size)
            deadline_unpermuted = base_batch + noise_batch
            permutations = torch.argsort(torch.rand(num_samples, size), dim=1)
            deadline = torch.gather(deadline_unpermuted, 1, permutations)  # Shape: (num_samples, size)


            source = torch.rand(num_samples, size, 2)

            operation_time = torch.empty(num_samples, size, paramet_hrsp.ROBOT_TYPE_NUM)

            for i in range(paramet_hrsp.ROBOT_TYPE_NUM):
                range_left = operation_range_list[i][0]
                range_right = operation_range_list[i][1]
                operation_time[..., i] = torch.rand(num_samples, size) * (range_right - range_left) + range_left



            self.data = []
            for i in range(num_samples):
                self.data.append({
                    'source': source[i],
                    'deadline': deadline[i],
                    'operation_time': operation_time[i],
                })


        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def make_instance(args):
    source, destination, deadline, required_robot, operation_time, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'source': torch.tensor(source, dtype=torch.float) / grid_size,
        'destination': torch.tensor(destination, dtype=torch.float) / grid_size,
        'deadline': torch.tensor(deadline, dtype=torch.float),
        'required_robot': torch.tensor(required_robot, dtype=torch.float),
        'operation_time': torch.tensor(operation_time, dtype=torch.float)
    }

