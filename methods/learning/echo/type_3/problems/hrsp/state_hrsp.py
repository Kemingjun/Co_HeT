import torch
from typing import NamedTuple, Dict
from utils.boolmask import mask_long2bool, mask_long_scatter
from problems.hrsp.paramet_hrsp import paramet_hrsp


class StateHRSP(NamedTuple):
    # Fixed input
    coords: torch.Tensor
    deadline: torch.Tensor
    operation_time: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows


    cur_time: torch.Tensor
    cur_coord: torch.Tensor
    visited_: torch.Tensor
    length: torch.Tensor
    tardiness: torch.Tensor
    i: torch.Tensor           # Keeps track of step

    VEHICLE_VELOCITY = 1.0


    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=1, dim=-1)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        coords = input['source']

        batch_size, n_loc, _ = coords.size()


        cur_time = torch.zeros(batch_size, paramet_hrsp.ROBOT_NUM, dtype=torch.float, device=coords.device)
        cur_coord = torch.zeros(batch_size, paramet_hrsp.ROBOT_NUM, 2, dtype=torch.float, device=coords.device)
        length = torch.zeros(batch_size, paramet_hrsp.ROBOT_NUM, dtype=torch.float, device=coords.device)
        tardiness = torch.zeros(batch_size, 1, dtype=torch.float, device=coords.device)



        return StateHRSP(
            coords=coords,
            deadline=input['deadline'],
            operation_time=input['operation_time'],
            ids=torch.arange(batch_size, dtype=torch.int64, device=coords.device)[:, None],  # Add steps dimension
            # prev_task=prev_task,
            cur_time=cur_time,
            cur_coord=cur_coord,
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=coords.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=coords.device)  # Ceil
            ),
            length=length,
            tardiness=tardiness,
            i=torch.zeros(1, dtype=torch.int64, device=coords.device)  # Vector with length num_steps
        )

    def update(self, selected_task, selected_robot_one_hot):

        """
        Args:
            selected_task: [batch_size]
            selected_robot_one_hot: [batch_size, robot_num]  true    false 

        Returns:

        """
        assert self.i.size(0) == 1, "Can only update if state represents single step"
        selected_robot_one_hot = selected_robot_one_hot.bool()

        # prev_task = self.prev_task.clone()
        # prev_task[selected_robot_one_hot] = (selected_task + 1).unsqueeze(-1).expand_as(selected_robot_one_hot)[selected_robot_one_hot]

        cur_coord = self.cur_coord.clone()
        # selected_task_coord = self.coords.gather(1, selected_task[:, None, None].expand(-1, 1, 2) - 1)
        selected_task_coord = self.coords[self.ids.squeeze(), selected_task]
        # expanded_coord = selected_task_coord.expand(-1, selected_robot.size(1), -1)
        cur_coord[selected_robot_one_hot] = selected_task_coord.unsqueeze(1).expand(-1, selected_robot_one_hot.size(1), -1)[selected_robot_one_hot]
        # cur_coord = torch.where(
        #     selected_robot[:, :, None],  # shape [batch_size, robot_num, 1]
        # )
        # cur_coord[selected_robot] = selected_task_coord[selected_robot]

        diff = cur_coord - self.cur_coord
        distance = diff.norm(p=1, dim=-1)
        length = self.length + distance

        travel_time = distance / paramet_hrsp.ROBOT_VELOCITY


        updated_time = self.cur_time.clone()
        updated_time[selected_robot_one_hot] += travel_time[selected_robot_one_hot]
        masked_time = updated_time.masked_fill(~selected_robot_one_hot, float('-inf'))
        max_time_per_batch, _ = masked_time.max(dim=1, keepdim=True)
        updated_time[selected_robot_one_hot] = max_time_per_batch.expand_as(updated_time)[selected_robot_one_hot]

        op_time_for_selected_task = self.operation_time[self.ids.squeeze(), selected_task]


        robot_type_indices = paramet_hrsp.robot_type_indices.to(updated_time.device)
        robot_op_time = op_time_for_selected_task.gather(1, robot_type_indices.expand(updated_time.size(0), -1))

        updated_time[selected_robot_one_hot] += robot_op_time[selected_robot_one_hot]

        masked_final_finish_times = updated_time.masked_fill(~selected_robot_one_hot, float('-inf'))
        complete_time, _ = masked_final_finish_times.max(dim=1)

        selected_task_deadline = self.deadline[self.ids.squeeze(), selected_task]

        selected_task_tardiness = torch.clamp_min(complete_time - selected_task_deadline, 0)

        # tardiness = torch.clamp_min(complete_time - self.deadline, 0)





        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, selected_task[:, None, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, selected_task[:, None])

        return self._replace(
            cur_time=updated_time, cur_coord=cur_coord, visited_=visited_,
            length=length, i=self.i + 1, tardiness=self.tardiness + selected_task_tardiness.unsqueeze(-1),
        )

    def all_finished(self):
        return self.i.item() >= self.coords.size(1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_task

    def get_mask(self):
        visited = self.visited_.to(torch.bool)
        return visited



