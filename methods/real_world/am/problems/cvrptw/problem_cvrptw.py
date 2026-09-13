from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.cvrptw.state_cvrptw import StateCVRPTW






class CVRPTW(object):

    NAME = 'cvrptw'  # Time-window Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0

    VEHICLE_VELOCITY = 1.0

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
                       torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
                       sorted_pi[:, -graph_size:]
               ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        '''检查解是否合法'''
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRPTW.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )  # [-1.0000,  0.0333,  0.2000,  ...,  0.0667,  0.1667,  0.1333]
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRPTW.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        ''' 计算时间 '''

        depot_col = torch.zeros(batch_size, 1, dtype=pi.dtype, device=pi.device)
        depot_pi = torch.cat([depot_col, pi], dim=1)  # 原本的pi，从第一个任务直接开始

        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)

        path_coords = loc_with_depot.gather(1, depot_pi[..., None].expand(*depot_pi.size(), loc_with_depot.size(-1)))

        travel_dist = (path_coords[:, 1:] - path_coords[:, :-1]).norm(p=2, dim=2)
        segment_times = travel_dist / CVRPTW.VEHICLE_VELOCITY

        arrival_times = torch.zeros_like(depot_pi, dtype=torch.float32)

        is_depot = (depot_pi == 0)

        deadline_with_depot = torch.cat(
            (
                torch.full_like(dataset['deadline'][:, :1], 10),
                dataset['deadline']
            ),
            1
        )  # [-1.0000,  0.0333,  0.2000,  ...,  0.0667,  0.1667,  0.1333]

        deadline = deadline_with_depot.gather(1, depot_pi)

        for i in range(1, depot_pi.size(1)):
            # 如果前一个节点是depot，则当前段的时间从0开始累积
            arrival_times[:, i] = torch.where(
                is_depot[:, i],
                0,  # 重置后，当前段的时间直接赋值（相当于从0开始加）
                arrival_times[:, i - 1] + segment_times[:, i - 1]  # 否则继续累积
            )
            assert (arrival_times[:, i] <= deadline[:, i]).all(), "Violate the time window constraint"

        return (
            travel_dist.sum(1)
            + (path_coords[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None



    @staticmethod
    def make_dataset(*args, **kwargs):
        return CVRPTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRPTW.initialize(*args, **kwargs)


class CVRPTWDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(CVRPTWDataset, self).__init__()

        self.data_set = []

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]

        else:

            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }

            # self.data = [
            #     {
            #         'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
            #         # Uniform 1 - 9, scaled by capacities
            #         'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
            #         'depot': torch.FloatTensor(2).uniform_(0, 1),  # depot随机生成
            #         'deadline': torch.FloatTensor(size).uniform_(1, 1.5)
            #     }
            #     for i in range(num_samples)
            # ]
            min_margin = 0.4
            max_margin = 0.8
            self.data = [
                (lambda loc, depot:
                 {
                     'loc': loc,
                     'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                     'depot': depot,
                     'deadline': ((loc - depot).norm(p=2, dim=1) + torch.FloatTensor(size).uniform_(min_margin, max_margin))
                 }
                 )(
                    torch.FloatTensor(size, 2).uniform_(0, 1),  # loc
                    torch.FloatTensor(2).uniform_(0, 1)  # depot
                )
                for _ in range(num_samples)
            ]


        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def make_instance(args):
    depot, loc, demand, capacity, deadline, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'deadline': torch.tensor(deadline, dtype=torch.float)
    }