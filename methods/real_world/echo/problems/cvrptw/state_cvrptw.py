import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCVRPTW(NamedTuple):
    # Fixed input
    coords: torch.Tensor
    demand: torch.Tensor
    deadline: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows


    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    cur_time: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    VEHICLE_CAPACITY = 1.0  # Hardcoded

    VEHICLE_VELOCITY = 1.0


    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
            cur_time=self.cur_time[key]
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['loc']
        demand = input['demand']
        deadline = input['deadline']

        batch_size, n_loc, _ = loc.size()
        return StateCVRPTW(
            coords=torch.cat((depot[:, None, :], loc), -2),
            demand=demand,
            deadline=deadline,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            cur_time=torch.zeros(batch_size, 1, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):
        assert self.i.size(0) == 1, "Can only update if state represents single step"
        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
        n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        cur_coord = self.coords[self.ids, selected]

        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]

        used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()

        is_depot = (selected == 0)

        travel_time = (cur_coord - self.cur_coord).norm(p=2, dim=-1, keepdim=True) / self.VEHICLE_VELOCITY  # (B, 1)
        travel_time = travel_time.reshape(-1, 1)
        cur_time = torch.where(
            is_depot,  # condition
            torch.zeros_like(self.cur_time),  # if depot, set 0
            self.cur_time + travel_time  # else, accumulate travel time
        )

        # cur_time = self.cur_time + (cur_coord - self.cur_coord).norm(p=2, dim=-1) / self.VEHICLE_VELOCITY if selected != 0 else 0


        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1, cur_time=cur_time
        )


    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)

        ''' 计算时间相关的mask '''
        travel_time = ((self.cur_coord - self.coords).norm(p=2, dim=2)) / self.VEHICLE_VELOCITY
        # 2. 到达时间 = 当前时间 + travel time
        arrival_time = self.cur_time + travel_time  # shape: (B, N)

        deadline_with_depot = torch.cat(
            (
                torch.full_like(self.deadline[:, :1], 10),
                self.deadline
            ),
            1
        )
        # 3. 超过 due_time 的位置设置为 True（需要 mask 掉）
        time_window_mask = arrival_time > deadline_with_depot  # shape: (B, N), bool

        time_window_mask = time_window_mask.unsqueeze(1)  # shape: (B, 1, N)


        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap

        # 前一步就是depot，且仍然有没有访问的节点
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)

        capacity_mask = torch.cat((mask_depot[:, :, None], mask_loc), -1)

        mask = time_window_mask | capacity_mask

        all_masked = mask.all(dim=2)  # shape: (B, 1)，每个样本是否全部为 True
        has_all_masked = all_masked.any()  # 是否存在“全是 True”的样本
        assert not has_all_masked, "all masked"

        return mask

    def construct_solutions(self, actions):
        return actions






