from typing import NamedTuple

import torch

from problems.hrsp.paramet_hrsp import paramet_hrsp
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateHRSP(NamedTuple):
    supply_raw: torch.Tensor
    handover_raw: torch.Tensor
    delivery_raw: torch.Tensor
    supply_norm: torch.Tensor
    handover_norm: torch.Tensor
    delivery_norm: torch.Tensor
    deadline: torch.Tensor
    required_robot: torch.Tensor
    ids: torch.Tensor
    cur_time: torch.Tensor
    carrier_pos_raw: torch.Tensor
    shuttle_pos_raw: torch.Tensor
    forklift_pos_raw: torch.Tensor
    carrier_pos_norm: torch.Tensor
    shuttle_pos_norm: torch.Tensor
    forklift_pos_norm: torch.Tensor
    length: torch.Tensor
    tardiness: torch.Tensor
    route: torch.Tensor
    route_len: torch.Tensor
    visited_: torch.Tensor
    i: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        return mask_long2bool(self.visited_, n=self.supply_raw.size(1))

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        supply_raw = input["supply_raw"]
        batch_size, n_loc, _ = supply_raw.size()
        device = supply_raw.device
        depot_raw = paramet_hrsp.DEPOT_RAW.to(device).view(1, 1, 2).expand(batch_size, -1, -1)
        depot_norm = paramet_hrsp.DEPOT_NORM.to(device).view(1, 1, 2).expand(batch_size, -1, -1)

        return StateHRSP(
            supply_raw=supply_raw,
            handover_raw=input["handover_raw"],
            delivery_raw=input["delivery_raw"],
            supply_norm=input["supply_norm"],
            handover_norm=input["handover_norm"],
            delivery_norm=input["delivery_norm"],
            deadline=input["deadline"],
            required_robot=input["required_robot"],
            ids=torch.arange(batch_size, dtype=torch.int64, device=device)[:, None],
            cur_time=torch.zeros(batch_size, paramet_hrsp.ROBOT_NUM, dtype=torch.float, device=device),
            carrier_pos_raw=depot_raw.expand(-1, int(paramet_hrsp.ROBOT_NUM_LIST[0].item()), -1).clone(),
            shuttle_pos_raw=depot_raw.expand(-1, int(paramet_hrsp.ROBOT_NUM_LIST[1].item()), -1).clone(),
            forklift_pos_raw=depot_raw.expand(-1, int(paramet_hrsp.ROBOT_NUM_LIST[2].item()), -1).clone(),
            carrier_pos_norm=depot_norm.expand(-1, int(paramet_hrsp.ROBOT_NUM_LIST[0].item()), -1).clone(),
            shuttle_pos_norm=depot_norm.expand(-1, int(paramet_hrsp.ROBOT_NUM_LIST[1].item()), -1).clone(),
            forklift_pos_norm=depot_norm.expand(-1, int(paramet_hrsp.ROBOT_NUM_LIST[2].item()), -1).clone(),
            length=torch.zeros(batch_size, 1, dtype=torch.float, device=device),
            tardiness=torch.zeros(batch_size, 1, dtype=torch.float, device=device),
            route=torch.full(
                (batch_size, paramet_hrsp.ROBOT_NUM, n_loc),
                -1,
                dtype=torch.long,
                device=device,
            ),
            route_len=torch.zeros(batch_size, paramet_hrsp.ROBOT_NUM, dtype=torch.long, device=device),
            visited_=(
                torch.zeros(batch_size, 1, n_loc, dtype=torch.uint8, device=device)
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=device)
            ),
            i=torch.zeros(1, dtype=torch.int64, device=device),
        )

    @property
    def cur_coord(self):
        return torch.cat([self.carrier_pos_norm, self.shuttle_pos_norm, self.forklift_pos_norm], dim=1)

    def update(self, selected_task, selected_robot_one_hot):
        selected_robot_one_hot = selected_robot_one_hot.bool()
        batch_idx = self.ids.squeeze()
        supply_raw = self.supply_raw[batch_idx, selected_task]
        handover_raw = self.handover_raw[batch_idx, selected_task]
        delivery_raw = self.delivery_raw[batch_idx, selected_task]
        handover_norm = self.handover_norm[batch_idx, selected_task]
        delivery_norm = self.delivery_norm[batch_idx, selected_task]

        c_n = int(paramet_hrsp.ROBOT_NUM_LIST[0].item())
        s_n = int(paramet_hrsp.ROBOT_NUM_LIST[1].item())
        c_mask = selected_robot_one_hot[:, :c_n]
        s_mask = selected_robot_one_hot[:, c_n:c_n + s_n]
        f_mask = selected_robot_one_hot[:, c_n + s_n:]

        carrier_idx = c_mask.float().argmax(dim=1)
        shuttle_idx = s_mask.float().argmax(dim=1)
        forklift_idx = f_mask.float().argmax(dim=1)
        global_shuttle_idx = shuttle_idx + c_n
        global_forklift_idx = forklift_idx + c_n + s_n

        carrier_pre_time = self.cur_time[batch_idx, carrier_idx]
        shuttle_pre_time = self.cur_time[batch_idx, global_shuttle_idx]
        forklift_pre_time = self.cur_time[batch_idx, global_forklift_idx]

        carrier_pre_pos = self.carrier_pos_raw[batch_idx, carrier_idx]
        shuttle_pre_pos = self.shuttle_pos_raw[batch_idx, shuttle_idx]
        forklift_pre_pos = self.forklift_pos_raw[batch_idx, forklift_idx]

        carrier_to_shuttle = manhattan(carrier_pre_pos, shuttle_pre_pos)
        shuttle_to_handover = manhattan(shuttle_pre_pos, handover_raw)
        handover_to_delivery = manhattan(handover_raw, delivery_raw)
        forklift_to_supply = manhattan(forklift_pre_pos, supply_raw)
        supply_to_handover = manhattan(supply_raw, handover_raw)

        carrier_arrive_ad = carrier_pre_time + carrier_to_shuttle / paramet_hrsp.CARRIER_VELOCITY
        attach_start = torch.maximum(carrier_arrive_ad, shuttle_pre_time)
        attach_end = attach_start + paramet_hrsp.CARRIER_SHUTTLE_COUPLING_TIME
        carrier_shuttle_arrive_handover = attach_end + shuttle_to_handover / paramet_hrsp.CARRIER_VELOCITY

        forklift_arrive_supply = forklift_pre_time + forklift_to_supply / paramet_hrsp.FORKLIFT_VELOCITY
        forklift_depart_supply = forklift_arrive_supply + paramet_hrsp.FORKLIFT_PICKUP_TIME
        forklift_arrive_handover = forklift_depart_supply + supply_to_handover / paramet_hrsp.FORKLIFT_VELOCITY

        handover_start = torch.maximum(carrier_shuttle_arrive_handover, forklift_arrive_handover)
        handover_end = handover_start + paramet_hrsp.SOURCE_HANDOVER_TIME
        forklift_release = handover_end
        arrive_destination = handover_end + handover_to_delivery / paramet_hrsp.CARRIER_VELOCITY
        carrier_release = arrive_destination + paramet_hrsp.CARRIER_SHUTTLE_DECOUPLING_TIME
        shuttle_release = (
            carrier_release
            + paramet_hrsp.SHUTTLE_UNLOADING_TIME
            + paramet_hrsp.DELIVERY_STATION_PROCESSING_TIME
        )

        cur_time = self.cur_time.clone()
        cur_time[batch_idx, carrier_idx] = carrier_release
        cur_time[batch_idx, global_shuttle_idx] = shuttle_release
        cur_time[batch_idx, global_forklift_idx] = forklift_release

        carrier_pos_raw = self.carrier_pos_raw.clone()
        shuttle_pos_raw = self.shuttle_pos_raw.clone()
        forklift_pos_raw = self.forklift_pos_raw.clone()
        carrier_pos_raw[batch_idx, carrier_idx] = delivery_raw
        shuttle_pos_raw[batch_idx, shuttle_idx] = delivery_raw
        forklift_pos_raw[batch_idx, forklift_idx] = handover_raw

        carrier_pos_norm = self.carrier_pos_norm.clone()
        shuttle_pos_norm = self.shuttle_pos_norm.clone()
        forklift_pos_norm = self.forklift_pos_norm.clone()
        carrier_pos_norm[batch_idx, carrier_idx] = delivery_norm
        shuttle_pos_norm[batch_idx, shuttle_idx] = delivery_norm
        forklift_pos_norm[batch_idx, forklift_idx] = handover_norm

        task_distance = carrier_to_shuttle + shuttle_to_handover + handover_to_delivery + forklift_to_supply + supply_to_handover
        selected_deadline = self.deadline[batch_idx, selected_task]
        task_tardiness = torch.clamp_min(shuttle_release - selected_deadline, 0)

        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, selected_task[:, None, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, selected_task[:, None])

        route = self.route.clone()
        route_len = self.route_len.clone()
        selected_b, selected_r = selected_robot_one_hot.nonzero(as_tuple=True)
        selected_pos = route_len[selected_b, selected_r]
        route[selected_b, selected_r, selected_pos] = selected_task[selected_b]
        route_len[selected_b, selected_r] = route_len[selected_b, selected_r] + 1

        return self._replace(
            cur_time=cur_time,
            carrier_pos_raw=carrier_pos_raw,
            shuttle_pos_raw=shuttle_pos_raw,
            forklift_pos_raw=forklift_pos_raw,
            carrier_pos_norm=carrier_pos_norm,
            shuttle_pos_norm=shuttle_pos_norm,
            forklift_pos_norm=forklift_pos_norm,
            length=self.length + task_distance.unsqueeze(-1),
            tardiness=self.tardiness + task_tardiness.unsqueeze(-1),
            route=route,
            route_len=route_len,
            visited_=visited_,
            i=self.i + 1,
        )

    def get_routes(self):
        routes = []
        for b in range(self.route.size(0)):
            batch_routes = []
            for r in range(self.route.size(1)):
                route_len = int(self.route_len[b, r].item())
                batch_routes.append(self.route[b, r, :route_len].tolist())
            routes.append(batch_routes)
        return routes

    def all_finished(self):
        return self.i.item() >= self.supply_raw.size(1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_mask(self):
        return self.visited_.to(torch.bool)


def manhattan(a, b):
    return (a - b).abs().sum(dim=-1)


